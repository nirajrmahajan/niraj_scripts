import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from helpers.csv_utils import ALL_ABN_KEYS, FP_REDBRICKS2CSV_MAP, _eval_if_str, _get_lung_bbox, validate_final_df
from safetensors.torch import load_file
from tqdm.auto import tqdm

CACHE_ROOT = "/raid26/niraj/storage/ctrate_cache"
MIN_CONF_THRESH = 0.7
MAX_FP_PER_SCAN = 20
MIN_CT_SLICES = 64


def _convert_bbox_from_tensor(tens):
    if isinstance(tens, dict):
        return tens
    z1, y1, x1, z2, y2, x2 = tens
    zc = ((z1 + z2) / 2).item()
    yc = ((y1 + y2) / 2).item()
    xc = ((x1 + x2) / 2).item()
    d = (z2 - z1).item()
    h = (y2 - y1).item()
    w = (x2 - x1).item()
    return {"zc": zc, "yc": yc, "xc": xc, "d": d, "h": h, "w": w}


def _load_single_pred_file(args):
    """Helper function to load a single prediction file - used for parallelization."""
    raw_preds_path, preds_file = args
    sid = os.path.basename(preds_file).replace(".pth", "")
    pred_data = torch.load(os.path.join(raw_preds_path, preds_file), weights_only=False)
    return sid, pred_data


def _get_all_preds_dict(num_workers=1):
    all_preds_dict = {}
    raw_preds_path = os.path.join(CACHE_ROOT, "nodule_detection_preds/preds_raw_sid_wise/")
    assert os.path.exists(raw_preds_path), f"Raw predictions path {raw_preds_path} does not exist"

    preds_files = os.listdir(raw_preds_path)
    filtered_df = pd.concat(
        [
            pd.read_csv(os.path.join(CACHE_ROOT, "metadata/multi_abnormality_labels/train_joined.csv")),
            pd.read_csv(os.path.join(CACHE_ROOT, "metadata/multi_abnormality_labels/valid_joined.csv")),
        ]
    )
    filtered_df = filtered_df[filtered_df["Lung nodule"] == 0]
    all_nodule_neg_sids = [x.replace(".nii.gz", "") for x in filtered_df.VolumeName.unique().tolist()]

    file_args = [
        (raw_preds_path, preds_file)
        for preds_file in preds_files
        if preds_file.replace(".pth", "") in all_nodule_neg_sids
    ]

    if num_workers == 1:
        # Simple for loop
        for args in tqdm(file_args, desc="Loading prediction files"):
            sid, pred_data = _load_single_pred_file(args)
            all_preds_dict[sid] = pred_data
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for args in tqdm(file_args, desc="Submitting prediction file tasks"):
                futures[executor.submit(_load_single_pred_file, args)] = args

            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading prediction files"):
                sid, pred_data = future.result()
                all_preds_dict[sid] = pred_data

    return {k: v for k, v in all_preds_dict.items() if k in all_nodule_neg_sids}


def _generate_all_abn_dicts_per_series_id(all_preds_dict):
    all_sids = all_preds_dict.keys()
    ans = {}
    for sid in all_sids:
        ans[sid] = {}
        boxes = all_preds_dict[sid]["boxes"]
        scores = all_preds_dict[sid]["scores"]
        assert len(boxes) == len(scores), f"Length of boxes and scores must be the same for series {sid}"
        filtered_boxes = [(b, s) for b, s in zip(boxes, scores) if s >= MIN_CONF_THRESH]
        # Keep only the highest confidence false positives per scan to cap annotation count.
        top_filtered_boxes = sorted(filtered_boxes, key=lambda x: x[1], reverse=True)[:MAX_FP_PER_SCAN]
        for i, (bbox, _) in enumerate(top_filtered_boxes):
            ans[sid]["non_nodule_fp"] = ans[sid].get("non_nodule_fp", []) + [
                (f"{sid}_fp_{i}", _convert_bbox_from_tensor(bbox))
            ]

    return ans


def _generate_master_df(all_detection_preds, all_abn_dicts_per_series_id, num_workers=1):
    # Simple for loop
    master_data = []
    for sid in tqdm(all_detection_preds.keys(), desc="Generating Master DF"):
        all_abn_dict = all_abn_dicts_per_series_id[sid]
        lung_bbox = _convert_bbox_from_tensor(all_detection_preds[sid]["lung_box"])
        spacing = all_detection_preds[sid]["spacing"]
        spacing = {"z": spacing[0].item(), "y": spacing[1].item(), "x": spacing[2].item()}
        if sid.startswith("train"):
            safetensor_path = os.path.join(CACHE_ROOT, "train", f"{sid}.safetensors")
        elif sid.startswith("valid"):
            safetensor_path = os.path.join(CACHE_ROOT, "valid", f"{sid}.safetensors")
        else:
            raise ValueError(f"Invalid sid: {sid}")
        if "non_nodule_fp" not in all_abn_dict:
            continue
        for annot_id, bbox in all_abn_dict["non_nodule_fp"]:
            bbox = _convert_bbox_from_tensor(bbox)
            master_data.append(
                {
                    "series_uid": sid,
                    "annot_id": annot_id,
                    "safetensor_path": safetensor_path,
                    "abn_bbox": bbox,
                    "lung_bbox": lung_bbox,
                    "all_abn_bboxes_in_sid": all_abn_dict,
                    "spacing": spacing,
                    "split": "train" if sid.startswith("train") else "valid_test",
                    "dataset_name": "ctrate_fp",
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
            )
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

    all_detection_preds = _get_all_preds_dict(num_workers=args.num_workers)

    all_abn_dicts_per_series_id = _generate_all_abn_dicts_per_series_id(all_detection_preds)

    master_df = _generate_master_df(all_detection_preds, all_abn_dicts_per_series_id, num_workers=args.num_workers)
    master_df = _drop_scans_with_few_slices(master_df)

    validate_final_df(master_df)
    print(f"Master dataframe validated")

    nested_cols = ["abn_bbox", "lung_bbox", "all_abn_bboxes_in_sid", "spacing"]
    master_df_to_save = master_df.copy()
    for col in nested_cols:
        if col in master_df_to_save.columns:
            tqdm.pandas(desc=f"Serializing {col}")
            master_df_to_save[col] = master_df_to_save[col].progress_apply(
                lambda x: json.dumps(x, default=str) if isinstance(x, (dict, list, tuple)) else x
            )

    master_df_path = os.path.join("ctrate_fp_master.parquet")

    master_df_to_save.to_parquet(master_df_path, index=False, engine="pyarrow")
    print(f"Master dataframe saved to {master_df_path}")
