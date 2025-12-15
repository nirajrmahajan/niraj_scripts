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
    mongo_collection_name = row["scan_filepath"].split("/")[-2]
    lung_mask_path = os.path.join(
        CACHE_ROOT, "annotations/lung_masks", mongo_collection_name, f"{row['series_uid']}.nii.gz"
    )
    if not os.path.exists(lung_mask_path):
        print(f"Lung mask not found for {row['series_uid']}")
        return None
    return _get_lung_bbox(lung_mask_path)


def _build_spacing(row):
    return {"z": row["spacing_z"], "y": row["spacing_y"], "x": row["spacing_x"]}


def _cache_lung_bboxes(df, num_workers=1):
    unique_rows = df.drop_duplicates(subset=["series_uid"])
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

    return {row["series_uid"]: bbox for row, bbox in zip(row_dicts, lung_bboxes)}


def _load_lung_bbox_cache_from_backup(backup_path1, backup_path2, unique_rows):
    if not os.path.exists(backup_path1) and not os.path.exists(backup_path2):
        return None

    global _BACKUP_LUNG_BBOX_CACHE
    global _BACKUP_LUNG_BBOX_CACHE_PATH

    if _BACKUP_LUNG_BBOX_CACHE_PATH == backup_path1 and _BACKUP_LUNG_BBOX_CACHE is not None:
        cache = _BACKUP_LUNG_BBOX_CACHE
    else:
        try:
            backup_df1 = pd.read_parquet(backup_path1, columns=["series_uid", "lung_bbox"])
            backup_df2 = pd.read_parquet(backup_path2, columns=["series_uid", "lung_bbox"])
            backup_df = pd.concat([backup_df1, backup_df2], ignore_index=True)
        except Exception as exc:
            print(f"Failed to load backup parquet {backup_path1} or {backup_path2}: {exc}")
            return None

        required_cols = {"series_uid", "lung_bbox"}
        if not required_cols.issubset(set(backup_df.columns)):
            print(
                f"Backup parquet {backup_path1} and {backup_path2} missing required columns {required_cols}; recomputing lung bboxes."
            )
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
            print(
                f"Backup parquet {backup_path1} and {backup_path2} did not yield any lung bboxes; recomputing from masks."
            )
            return None

        _BACKUP_LUNG_BBOX_CACHE = cache
        _BACKUP_LUNG_BBOX_CACHE_PATH = backup_path1

    required_keys = set(unique_rows["series_uid"])
    missing = list(required_keys - set(cache.keys()))
    print(f"Loaded lung bboxes for {len(cache)} series from backup {backup_path1} and {backup_path2}.")
    if missing:
        print(f"Backup missing lung bboxes for {len(missing)} series; will compute those from masks.")
    return cache.copy(), missing


def _generate_all_abn_dicts_per_series_id(test_fp_df, aid2reason_fp):
    all_preds_dict = {}
    for i in tqdm(range(len(test_fp_df)), "Generating Abnormality Dictionaries"):
        row = test_fp_df.iloc[i]
        series_uid = row["series_uid"]
        annot_id = row["annot_uid"]
        fp_category = {"2d": "nodule", "2c": "non_nodule_fp", "FN": "fn_nodule"}[row["category"]]
        if fp_category == "fn_nodule":
            continue
        z1, y1, x1, z2, y2, x2 = _eval_if_str(row["bbox"])
        fp_bbox = {
            "z1": z1,
            "y1": y1,
            "x1": x1,
            "z2": z2,
            "y2": y2,
            "x2": x2,
        }
        if series_uid not in all_preds_dict:
            all_preds_dict[series_uid] = {}
        if fp_category not in all_preds_dict[series_uid]:
            all_preds_dict[series_uid][fp_category] = []
        all_preds_dict[series_uid][fp_category].append((annot_id, fp_bbox))
        if fp_category == "non_nodule_fp":
            reason_fp = aid2reason_fp.get(annot_id)
            if reason_fp == "pulmonary_vessel":
                all_preds_dict[series_uid]["pulmonary_vessel"] = all_preds_dict[series_uid].get(
                    "pulmonary_vessel", []
                ) + [(annot_id, fp_bbox)]
    return all_preds_dict


def _generate_master_df(test_fp_df, lung_bbox_map, sid2spacing, aid2reason_fp):
    # Simple for loop
    master_data = []
    for sid in tqdm(all_abn_dicts_per_series_id.keys(), desc="Generating Master DF"):
        all_abn_dict = all_abn_dicts_per_series_id[sid]
        test_fp_row = test_fp_df.loc[sid]
        if isinstance(test_fp_row, pd.DataFrame):
            test_fp_row = test_fp_row.iloc[0]
        data_split = test_fp_row["data_split"]

        lung_bbox = lung_bbox_map.get(sid)
        safetensor_path = test_fp_row["scan_filepath"]
        if sid not in sid2spacing:
            print(f"Spacing not found for {sid}")
            continue
        spacing = sid2spacing[sid]
        for abn_key in all_abn_dict.keys():
            for annot_id, bbox in all_abn_dict[abn_key]:
                if abn_key == "non_nodule_fp":
                    # check if vessel - if yes, skip - will be added by pulmonary vessel
                    reason_fp = aid2reason_fp.get(annot_id)
                    if reason_fp == "pulmonary_vessel":
                        continue
                    else:
                        # add as non nodule fp
                        dicc = {
                            "series_uid": sid,
                            "annot_id": annot_id,
                            "safetensor_path": safetensor_path,
                            "abn_bbox": bbox,
                            "lung_bbox": lung_bbox,
                            "all_abn_bboxes_in_sid": all_abn_dict,
                            "spacing": spacing,
                            "split": data_split,
                            "dataset_name": "airrc_lidc_fps",
                            "nodule": 0,
                            "fibrosis": -100,
                            "apical_pleural_thickening": -100,
                            "costochondral_junction_calcification": -100,
                            "osteophyte": -100,
                            "pulmonary_vessel": -100,
                            "ggo": -100,
                            "consolidation": -100,
                            "pleural_pathology": -100,
                            "fissure": -100,
                            "pleural_effusion": -100,
                            "intrapulmonary_lymph_node": -100,
                            "atelectasis": -100,
                            "reticulation": -100,
                            "other_fp": -100,
                            "non_nodule_fp": 1,
                        }
                elif abn_key == "nodule":
                    # add as nodule
                    dicc = {
                        "series_uid": sid,
                        "annot_id": annot_id,
                        "safetensor_path": safetensor_path,
                        "abn_bbox": bbox,
                        "lung_bbox": lung_bbox,
                        "all_abn_bboxes_in_sid": all_abn_dict,
                        "spacing": spacing,
                        "split": data_split,
                        "dataset_name": "airrc_lidc_fps",
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
                elif abn_key == "pulmonary_vessel":
                    # add as vessel
                    dicc = {
                        "series_uid": sid,
                        "annot_id": annot_id,
                        "safetensor_path": safetensor_path,
                        "abn_bbox": bbox,
                        "lung_bbox": lung_bbox,
                        "all_abn_bboxes_in_sid": all_abn_dict,
                        "spacing": spacing,
                        "split": data_split,
                        "dataset_name": "airrc_lidc_fps",
                        "nodule": 0,
                        "fibrosis": 0,
                        "apical_pleural_thickening": 0,
                        "costochondral_junction_calcification": 0,
                        "osteophyte": 0,
                        "pulmonary_vessel": 1,
                        "ggo": 0,
                        "consolidation": 0,
                        "pleural_pathology": 0,
                        "fissure": 0,
                        "pleural_effusion": 0,
                        "intrapulmonary_lymph_node": 0,
                        "atelectasis": 0,
                        "reticulation": 0,
                        "other_fp": 0,
                        "non_nodule_fp": 1,
                    }
                else:
                    raise ValueError(f"Invalid abn key: {abn_key}")

                master_data.append(dicc)

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

    test_fp_df = pd.read_csv(
        "/raid18/nirman_internal/detection_fpr/lidc_csv/lidc_allSplits_AirRC254_with_vessel_fps.csv"
    )
    test_fp_df.set_index("series_uid", inplace=True, drop=False)
    test_fp_df["scan_filepath"] = test_fp_df["scan_filepath"].replace(
        "qct_cache_august_2025", "qct_data_releases/qct_cache_october_2025", regex=True
    )
    aid2reason_fp = dict(zip(test_fp_df["annot_uid"], test_fp_df["reason_for_false_positive"]))

    unique_lung_rows = test_fp_df[["series_uid"]].drop_duplicates()
    backup_master_path1 = os.path.join("qct_cache_test_fp_master.parquet.bkp")
    backup_master_path2 = os.path.join("qct_cache_master.parquet.bkp")
    backup_result = _load_lung_bbox_cache_from_backup(backup_master_path1, backup_master_path2, unique_lung_rows)

    cache_master_df = pd.read_csv(os.path.join(CACHE_ROOT, "master_csv.csv"))
    sid2spacingx = dict(zip(cache_master_df["series_uid"], cache_master_df["spacing_x"]))
    sid2spacingy = dict(zip(cache_master_df["series_uid"], cache_master_df["spacing_y"]))
    sid2spacingz = dict(zip(cache_master_df["series_uid"], cache_master_df["spacing_z"]))
    sid2spacing = {}
    for sid in test_fp_df["series_uid"]:
        if sid not in sid2spacingx or sid not in sid2spacingy or sid not in sid2spacingz:
            print(f"Spacing not found for {sid}")
            continue
        sid2spacing[sid] = {
            "x": sid2spacingx[sid],
            "y": sid2spacingy[sid],
            "z": sid2spacingz[sid],
        }

    if backup_result is None:
        lung_bbox_map = _cache_lung_bboxes(test_fp_df, num_workers=args.num_workers)
    else:
        lung_bbox_map, missing_keys = backup_result
        if missing_keys:
            missing_set = set(missing_keys)
            mask = [sid in missing_set for sid in test_fp_df["series_uid"]]
            missing_df = test_fp_df[mask]
            if not missing_df.empty:
                regenerated = _cache_lung_bboxes(missing_df, num_workers=args.num_workers)
                lung_bbox_map.update(regenerated)

    print("Generating all abn dicts per series id")
    all_abn_dicts_per_series_id = _generate_all_abn_dicts_per_series_id(test_fp_df, aid2reason_fp)

    master_df = _generate_master_df(test_fp_df, lung_bbox_map, sid2spacing, aid2reason_fp)
    master_df = _drop_scans_with_few_slices(master_df)

    nested_cols = ["abn_bbox", "lung_bbox", "all_abn_bboxes_in_sid", "spacing"]
    master_df_to_save = master_df.copy()
    for col in nested_cols:
        if col in master_df_to_save.columns:
            tqdm.pandas(desc=f"Serializing {col}")
            master_df_to_save[col] = master_df_to_save[col].progress_apply(
                lambda x: json.dumps(x, default=str) if isinstance(x, (dict, list, tuple)) else x
            )

    master_df_path = os.path.join("airrc_fp_master.parquet")

    master_df_to_save.to_parquet(master_df_path, index=False, engine="pyarrow")
    print(f"Master dataframe saved to {master_df_path}")
    validate_final_df(master_df)
    print(f"Master dataframe validated")
    print(f"Master dataframe validated")
