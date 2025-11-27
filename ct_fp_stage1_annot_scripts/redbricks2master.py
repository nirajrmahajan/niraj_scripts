import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from helpers.csv_utils import (
    ALL_ABN_KEYS,
    FP_REDBRICKS2CSV_MAP,
    _add_data_split_column,
    _eval_if_str,
    _get_lung_bbox,
    validate_final_df,
)
from safetensors.torch import load_file
from tqdm.auto import tqdm

CACHE_ROOT = "/raid13/vikas/storage/qct_testing_cache"
MIN_CT_SLICES = 64
_WORKER_ABN_DICTS = None
_WORKER_LUNG_BBOXES = None
_BACKUP_LUNG_BBOX_CACHE = None
_BACKUP_LUNG_BBOX_CACHE_PATH = None


def _get_abn_key_from_df_meta(row):
    if _eval_if_str(row["meta"])["nodule_type"] == "true positive":
        return "nodule"
    elif _eval_if_str(row["meta"])["nodule_type"] == "false positive":
        fp_type = FP_REDBRICKS2CSV_MAP[_eval_if_str(row["meta"])["reason_for_false_positive"]]
        assert fp_type in ALL_ABN_KEYS, f"Invalid false positive type: {fp_type}. Must be one of {ALL_ABN_KEYS}"
        return fp_type
    else:
        raise ValueError(f"Invalid nodule type: {_eval_if_str(row['meta'])['nodule_type']}")


def _generate_all_abn_dicts_per_series_id(df):
    all_sids = df.series_uid.unique()
    ans = {}
    for sid in all_sids:
        sid_df = df[df.series_uid == sid]
        ans[sid] = {}
        for _, row in sid_df.iterrows():
            fp_type = _get_abn_key_from_df_meta(row)
            ans[sid][fp_type] = ans[sid].get(fp_type, []) + [(row["annot_id"], _eval_if_str(row["annot"])["bbox"])]
            if fp_type != "nodule":
                ans[sid]["non_nodule_fp"] = ans[sid].get("non_nodule_fp", []) + [
                    (row["annot_id"], _eval_if_str(row["annot"])["bbox"])
                ]
    return ans


def _cache_lung_bboxes(series_uids):
    cache = {}
    unique_sids = pd.unique(series_uids)
    for sid in tqdm(unique_sids, desc="Caching lung bboxes"):
        lung_mask_path = os.path.join(CACHE_ROOT, "lung_masks", "lung_masks", "penrad", f"{sid}.nii.gz")
        if not os.path.exists(lung_mask_path):
            continue
        cache[sid] = _get_lung_bbox(lung_mask_path)
    return cache


def _load_lung_bbox_cache_from_backup(backup_path, required_series_uids):
    if not os.path.exists(backup_path):
        return None

    global _BACKUP_LUNG_BBOX_CACHE
    global _BACKUP_LUNG_BBOX_CACHE_PATH

    if _BACKUP_LUNG_BBOX_CACHE_PATH == backup_path and _BACKUP_LUNG_BBOX_CACHE is not None:
        cache = _BACKUP_LUNG_BBOX_CACHE
    else:
        try:
            backup_df = pd.read_parquet(backup_path, columns=["series_uid", "lung_bbox"])
        except Exception as exc:
            print(f"Failed to load backup parquet {backup_path}: {exc}")
            return None

        required_cols = {"series_uid", "lung_bbox"}
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
        for sid, bbox in zip(backup_df["series_uid"], backup_df["lung_bbox"]):
            if sid in cache:
                continue
            parsed_bbox = _deserialize_bbox(bbox)
            if parsed_bbox is not None:
                cache[sid] = parsed_bbox

        if not cache:
            print(f"Backup parquet {backup_path} did not yield any lung bboxes; recomputing from masks.")
            return None

        _BACKUP_LUNG_BBOX_CACHE = cache
        _BACKUP_LUNG_BBOX_CACHE_PATH = backup_path

    required_set = set(pd.unique(required_series_uids))
    missing = list(required_set - set(cache.keys()))
    print(f"Loaded lung bboxes for {len(cache)} series from backup {backup_path}.")
    if missing:
        print(f"Backup missing lung bboxes for {len(missing)} new series; will compute those from masks.")
    return cache.copy(), missing


def _init_master_worker(all_abn_dicts_per_series_id, lung_bbox_cache):
    global _WORKER_ABN_DICTS
    global _WORKER_LUNG_BBOXES
    _WORKER_ABN_DICTS = all_abn_dicts_per_series_id
    _WORKER_LUNG_BBOXES = lung_bbox_cache


def _build_master_row_entry(row, all_abn_dicts_per_series_id, lung_bbox_cache):
    sid = row["series_uid"]
    all_abn_dict = all_abn_dicts_per_series_id[sid]
    lung_bbox = lung_bbox_cache.get(sid)
    if lung_bbox is None:
        return None

    safetensor_path = os.path.join(CACHE_ROOT, "full_resolution", "penrad", f"{sid}.safetensor")
    fp_type = _get_abn_key_from_df_meta(row)
    fp_dict = {
        "nodule": 0,
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
    assert fp_type in fp_dict, f"Invalid false positive type: {fp_type}. Must be one of {fp_dict.keys()}"
    fp_dict[fp_type] = 1
    if fp_type != "nodule":
        fp_dict["non_nodule_fp"] = 1

    row_dict = {
        "series_uid": sid,
        "annot_id": row["annot_id"],
        "safetensor_path": safetensor_path,
        "abn_bbox": _eval_if_str(row["annot"])["bbox"],
        "all_abn_bboxes_in_sid": all_abn_dict,
        "spacing": _eval_if_str(row["annot"])["spacing"],
        "lung_bbox": lung_bbox,
        "dataset_name": "penrad_stage1_fp",
    }
    row_dict.update(fp_dict)
    return row_dict


def _process_row_parallel(row_dict):
    if _WORKER_ABN_DICTS is None:
        raise RuntimeError("Parallel worker dictionary not initialized")
    if _WORKER_LUNG_BBOXES is None:
        raise RuntimeError("Parallel worker lung bbox cache not initialized")
    return _build_master_row_entry(row_dict, _WORKER_ABN_DICTS, _WORKER_LUNG_BBOXES)


def _generate_master_df(df, all_abn_dicts_per_series_id, lung_bbox_cache, num_workers=1):
    if num_workers == 1:
        master_data = []
        for i in tqdm(range(len(df)), desc="Generating Master DF"):
            row = df.iloc[i]
            master_entry = _build_master_row_entry(row, all_abn_dicts_per_series_id, lung_bbox_cache)
            if master_entry is not None:
                master_data.append(master_entry)
        return pd.DataFrame(master_data)

    rows = df.to_dict("records")
    master_data = []
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_master_worker,
        initargs=(all_abn_dicts_per_series_id, lung_bbox_cache),
    ) as executor:
        for result in tqdm(
            executor.map(_process_row_parallel, rows), total=len(rows), desc="Generating Master DF (parallel)"
        ):
            if result is not None:
                master_data.append(result)

    return pd.DataFrame(master_data)


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

    redbricks_csv = "/cache/fast_data_nas8/qct_segmentations/annotations/download/qCT_Lung_Cancer_penrad_6_stage_1_0925/annotations.csv"
    redbricks_df = pd.read_csv(redbricks_csv)

    all_abn_dicts_per_series_id = _generate_all_abn_dicts_per_series_id(redbricks_df)

    backup_master_path = os.path.join("redbricks_fp_master.parquet.bkp")
    backup_result = _load_lung_bbox_cache_from_backup(backup_master_path, redbricks_df["series_uid"])
    if backup_result is None:
        lung_bbox_cache = _cache_lung_bboxes(redbricks_df["series_uid"])
    else:
        lung_bbox_cache, missing_series = backup_result
        if missing_series:
            regenerated_cache = _cache_lung_bboxes(missing_series)
            lung_bbox_cache.update(regenerated_cache)

    master_df = _generate_master_df(
        redbricks_df, all_abn_dicts_per_series_id, lung_bbox_cache, num_workers=args.num_workers
    )
    master_df = _drop_scans_with_few_slices(master_df)

    master_df_with_split = _add_data_split_column(master_df.copy())
    validate_final_df(master_df_with_split)
    print("Master dataframe validated")

    nested_cols = ["abn_bbox", "lung_bbox", "all_abn_bboxes_in_sid", "spacing"]
    master_df_serialized = master_df_with_split.copy()
    for col in nested_cols:
        if col in master_df_serialized.columns:
            tqdm.pandas(desc=f"Serializing {col}")
            master_df_serialized[col] = master_df_serialized[col].progress_apply(
                lambda x: json.dumps(x, default=str) if isinstance(x, (dict, list, tuple)) else x
            )

    master_df_path = os.path.join("redbricks_fp_master.parquet")
    master_df_serialized.to_parquet(master_df_path, index=False, engine="pyarrow")
    print(f"Master dataframe saved to {master_df_path}")
