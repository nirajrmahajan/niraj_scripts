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

MIN_CONF_THRESH = 0.6
MAX_CONF_THRESH = 0.75
# MAX_FP_PER_SCAN = 20
MIN_CT_SLICES = 64
MIN_NODULE_BBOX_VOLUME = 250


def _generate_all_abn_dicts_per_series_id(qct_fp_df):
    all_preds_dict = {}
    for i in tqdm(range(len(qct_fp_df)), "Generating Abnormality Dictionaries"):
        row = qct_fp_df.iloc[i]
        series_uid = row["series_uid"]
        annot_id = row["annot_id"]
        score = row["nodule_bbox_conf"]
        fp_bbox = _eval_if_str(row["nodule_bbox"])
        if series_uid not in all_preds_dict:
            all_preds_dict[series_uid] = {}
        if "non_nodule_fp" not in all_preds_dict[series_uid]:
            all_preds_dict[series_uid]["non_nodule_fp"] = []
        all_preds_dict[series_uid]["non_nodule_fp"].append((annot_id, fp_bbox))
    return all_preds_dict


def _generate_master_df(qct_fp_df, all_abn_dicts_per_series_id, qct_nodule_master_df):
    # Simple for loop
    master_data = []
    for sid in tqdm(all_abn_dicts_per_series_id.keys(), desc="Generating Master DF"):
        all_abn_dict = all_abn_dicts_per_series_id[sid]
        nodule_tp_row = qct_nodule_master_df.loc[sid]
        if isinstance(nodule_tp_row, pd.DataFrame):
            nodule_tp_row = nodule_tp_row.iloc[0]

        lung_bbox = _eval_if_str(nodule_tp_row["lung_bbox"])
        safetensor_path = nodule_tp_row["safetensor_path"]
        spacing = _eval_if_str(nodule_tp_row["spacing"])
        for annot_id, bbox in all_abn_dict["non_nodule_fp"]:
            master_data.append(
                {
                    "series_uid": sid,
                    "annot_id": annot_id,
                    "safetensor_path": safetensor_path,
                    "abn_bbox": bbox,
                    "lung_bbox": lung_bbox,
                    "all_abn_bboxes_in_sid": all_abn_dict,
                    "spacing": spacing,
                    "split": "train",
                    "dataset_name": "qct_cache_fps",
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

    qct_nodule_master_df = pd.read_parquet(
        "/home/users/niraj.mahajan/projects/scripts/ct_fp_stage1_annot_scripts/qct_cache_master.parquet"
    )
    qct_nodule_master_df.set_index("series_uid", inplace=True, drop=False)
    qct_fp_df = pd.read_csv(
        "/raid13/niraj/storage/training/qct_training_framework/seresnet18_64_224_224_det_maxmip_spacing_finetune/checkpoints/finetune_dfs/fp_df.csv"
    )
    qct_fp_df = qct_fp_df[
        (qct_fp_df.series_uid.isin(qct_nodule_master_df.series_uid.unique()))
        & (qct_fp_df.nodule_bbox_conf >= MIN_CONF_THRESH)
        & (qct_fp_df.nodule_bbox_conf <= MAX_CONF_THRESH)
        & (qct_fp_df.nodule_bbox_volume_mm3 >= MIN_NODULE_BBOX_VOLUME)
    ]

    all_abn_dicts_per_series_id = _generate_all_abn_dicts_per_series_id(qct_fp_df)

    master_df = _generate_master_df(qct_fp_df, all_abn_dicts_per_series_id, qct_nodule_master_df)
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

    master_df_path = os.path.join("qct_cache_fp_master.parquet")

    master_df_to_save.to_parquet(master_df_path, index=False, engine="pyarrow")
    print(f"Master dataframe saved to {master_df_path}")
