import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from helpers.csv_utils import ALL_ABN_KEYS, FP_REDBRICKS2CSV_MAP, _eval_if_str, _get_lung_bbox, validate_final_df
from safetensors.torch import load_file
from tqdm.auto import tqdm

CACHE_ROOT = "/raid13/niraj/storage/qct_data_releases/qct_cache_october_2025/"
MIN_CT_SLICES = 64
_BACKUP_LUNG_BBOX_CACHE = None
_BACKUP_LUNG_BBOX_CACHE_PATH = None


def _format_all_abn_dict(raw_abn_dict):
    parsed = _eval_if_str(raw_abn_dict)
    if not parsed:
        return None
    return {"nodule": [(k, v) for k, v in parsed.items()]}


def _build_lung_bbox(row):
    lung_mask_path = os.path.join(
        CACHE_ROOT, "annotations/lung_masks", row["mongo_collection_name"], f"{row['series_uid']}.nii.gz"
    )
    if not os.path.exists(lung_mask_path):
        return None
    return _get_lung_bbox(lung_mask_path)


def _build_spacing(row):
    return {"z": row["spacing_z"], "y": row["spacing_y"], "x": row["spacing_x"]}


def _cache_lung_bboxes(df, num_workers=1):
    unique_rows = df[["series_uid", "mongo_collection_name"]].drop_duplicates()
    row_dicts = unique_rows.to_dict(orient="records")

    if num_workers == 1:
        tqdm.pandas(desc="Loading lung bboxes")
        lung_bboxes = [_build_lung_bbox(r) for r in tqdm(row_dicts, desc="Loading lung bboxes")]
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            lung_bboxes = list(
                tqdm(
                    executor.map(_build_lung_bbox, row_dicts),
                    total=len(row_dicts),
                    desc="Loading lung bboxes (mp)",
                )
            )

    return {(row["series_uid"], row["mongo_collection_name"]): bbox for row, bbox in zip(row_dicts, lung_bboxes)}


def _load_lung_bbox_cache_from_backup(backup_path, unique_rows):
    if not os.path.exists(backup_path):
        return None

    global _BACKUP_LUNG_BBOX_CACHE
    global _BACKUP_LUNG_BBOX_CACHE_PATH

    if _BACKUP_LUNG_BBOX_CACHE_PATH == backup_path and _BACKUP_LUNG_BBOX_CACHE is not None:
        cache = _BACKUP_LUNG_BBOX_CACHE
    else:
        try:
            backup_df = pd.read_parquet(backup_path, columns=["series_uid", "mongo_collection_name", "lung_bbox"])
        except Exception as exc:
            print(f"Failed to load backup parquet {backup_path}: {exc}")
            return None

        required_cols = {"series_uid", "mongo_collection_name", "lung_bbox"}
        if not required_cols.issubset(set(backup_df.columns)):
            print(f"Backup parquet {backup_path} missing required columns {required_cols}; recomputing lung bboxes.")
            return None

        def _deserialize_bbox(value):
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return None
            return None

        cache = {}
        for sid, coll, bbox in zip(backup_df["series_uid"], backup_df["mongo_collection_name"], backup_df["lung_bbox"]):
            key = (sid, coll)
            if key in cache:
                continue
            parsed_bbox = _deserialize_bbox(bbox)
            if parsed_bbox is not None:
                cache[key] = parsed_bbox

        if not cache:
            print(f"Backup parquet {backup_path} did not yield any lung bboxes; recomputing from masks.")
            return None

        _BACKUP_LUNG_BBOX_CACHE = cache
        _BACKUP_LUNG_BBOX_CACHE_PATH = backup_path

    required_keys = set(map(tuple, unique_rows.to_numpy()))
    missing = list(required_keys - set(cache.keys()))
    print(f"Loaded lung bboxes for {len(cache)} series/collections from backup {backup_path}.")
    if missing:
        print(f"Backup missing lung bboxes for {len(missing)} series/collections; will compute those from masks.")
    return cache.copy(), missing


def _generate_master_df(master_df, lung_bbox_map):
    df = master_df.copy()
    tqdm.pandas(desc="Parsing nodules")
    df["abn_bbox"] = df["nodule_bbox"].progress_apply(_eval_if_str)
    df["all_abn_bboxes_in_sid"] = df["nodule_all_bboxes_in_sid"].progress_apply(_format_all_abn_dict)
    df["safetensor_path"] = df.apply(
        lambda r: os.path.join(
            CACHE_ROOT, "full_resolution", r["mongo_collection_name"], f"{r['series_uid']}.safetensor"
        ),
        axis=1,
    )
    df["spacing"] = df.apply(_build_spacing, axis=1)

    df["lung_key"] = list(zip(df["series_uid"], df["mongo_collection_name"]))
    df["lung_bbox"] = df["lung_key"].map(lung_bbox_map)
    df = df.drop(columns=["lung_key"])
    df["dataset_name"] = "qct_annotated_data"
    df["split"] = df["data_split"]

    df = df.dropna(subset=["all_abn_bboxes_in_sid", "lung_bbox"])

    label_defaults = {
        "nodule": 1,
        "fibrosis": 0,
        "apical_pleural_thickening": 0,
        "costochondral_junction_calcification": 0,
        "osteophyte": 0,
        "pulmonary_vessel": 0,
        "ggo": 0,
        "consolidation": 0,
        "pleural_pathology": 0,
        "fissure": 0,
        "pleural_effusion": 0,
        "intrapulmonary_lymph_node": 0,
        "atelectasis": 0,
        "reticulation": 0,
        "other_fp": 0,
        "non_nodule_fp": 0,
    }
    for col, val in label_defaults.items():
        df[col] = val

    keep_cols = [
        "series_uid",
        "annot_id",
        "safetensor_path",
        "abn_bbox",
        "all_abn_bboxes_in_sid",
        "spacing",
        "lung_bbox",
        "dataset_name",
        "split",
        "mongo_collection_name",
    ] + list(label_defaults.keys())

    return df[keep_cols].reset_index(drop=True)


def _drop_scans_with_few_slices(df, min_slices=MIN_CT_SLICES):
    if df.empty:
        return df

    valid_paths = {}
    for path in tqdm(pd.unique(df["safetensor_path"]), desc=f"Filtering CTs < {min_slices} slices"):
        assert os.path.exists(path), f"Missing safetensor file: {path}"
        tensor_dict = load_file(path)
        assert "ct_data" in tensor_dict, f"'ct_data' not found in safetensor file {path}"
        shape = tensor_dict["ct_data"].shape
        assert len(shape) >= 3, f"Unexpected ct_data shape {shape} in {path}"
        num_slices = shape[-3]
        valid_paths[path] = num_slices >= min_slices

    filtered_df = df[df["safetensor_path"].map(valid_paths)].reset_index(drop=True)
    dropped = len(df) - len(filtered_df)
    if dropped:
        print(f"Dropped {dropped} annotations with CT depth < {min_slices}")
    else:
        print(f"All CT volumes have at least {min_slices} slices")
    return filtered_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate master dataframe from redbricks CSV")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel processing (default: 1)",
    )
    args = parser.parse_args()

    master_df = pd.read_csv(os.path.join(CACHE_ROOT, "master_csv.csv"))

    master_df = master_df[
        (master_df["scan_annot_type"].isin(["full_annot"]))
        & (master_df["annotated_by"] == "merged_annot")
        & (
            ~master_df["mongo_collection_name"].isin(
                [
                    "dedomena_non_cancer",
                    "qure_internal",
                    "deeplesion",
                    "penrad",
                    "lndb",
                    "qidw",
                    "aarthi_scans",
                    "wcg_fda_qct",
                ]
            )
        )
        & (master_df["nodule_num_rads_annotated_scan"] >= 2)
        & (master_df["nodule_long_axis_diameter"] >= 6)
        & (master_df["nodule_volume"] >= 60)
        & (master_df["spacing_z"] <= 5)
    ]

    unique_lung_rows = master_df[["series_uid", "mongo_collection_name"]].drop_duplicates()
    backup_master_path = os.path.join("qct_cache_master.parquet.bkp")
    backup_result = _load_lung_bbox_cache_from_backup(backup_master_path, unique_lung_rows)
    if backup_result is None:
        lung_bbox_map = _cache_lung_bboxes(master_df, num_workers=args.num_workers)
    else:
        lung_bbox_map, missing_keys = backup_result
        if missing_keys:
            missing_set = set(missing_keys)
            mask = [
                (sid, coll) in missing_set
                for sid, coll in zip(master_df["series_uid"], master_df["mongo_collection_name"])
            ]
            missing_df = master_df[mask]
            if not missing_df.empty:
                regenerated = _cache_lung_bboxes(missing_df, num_workers=args.num_workers)
                lung_bbox_map.update(regenerated)

    master_df = _generate_master_df(master_df, lung_bbox_map)
    master_df = _drop_scans_with_few_slices(master_df)

    nested_cols = ["abn_bbox", "lung_bbox", "all_abn_bboxes_in_sid", "spacing"]
    master_df_to_save = master_df.copy()
    for col in nested_cols:
        if col in master_df_to_save.columns:
            tqdm.pandas(desc=f"Serializing {col}")
            master_df_to_save[col] = master_df_to_save[col].progress_apply(
                lambda x: json.dumps(x, default=str) if isinstance(x, (dict, list, tuple)) else x
            )

    master_df_path = os.path.join("qct_cache_master.parquet")

    master_df_to_save.to_parquet(master_df_path, index=False, engine="pyarrow")
    print(f"Master dataframe saved to {master_df_path}")
    validate_final_df(master_df)
    print(f"Master dataframe validated")
