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

HIPAS_ROOT = "/raid13/niraj/storage/ct_ssl_downstream_datas/images_safetensor_dump/hipas"
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


def _generate_all_abn_dicts_per_series_id(vessel_summary_df):
    all_preds_dict = {}
    for i in range(len(vessel_summary_df)):
        row = vessel_summary_df.iloc[i]
        series_uid = str(row["series_uid"]).zfill(3)
        vessel_fp_bboxes = _eval_if_str(row["vessel_fp_bboxes"])
        if not vessel_fp_bboxes:
            continue
        if series_uid not in all_preds_dict:
            all_preds_dict[series_uid] = {}
        if "pulmonary_vessel" not in all_preds_dict[series_uid]:
            all_preds_dict[series_uid]["pulmonary_vessel"] = []
        if "non_nodule_fp" not in all_preds_dict[series_uid]:
            all_preds_dict[series_uid]["non_nodule_fp"] = []
        for ii, bbox in enumerate(vessel_fp_bboxes):
            z1, y1, x1, z2, y2, x2 = bbox
            all_preds_dict[series_uid]["pulmonary_vessel"].append(
                (f"{series_uid}_fp_{ii}", {"z1": z1, "y1": y1, "x1": x1, "z2": z2, "y2": y2, "x2": x2})
            )
            all_preds_dict[series_uid]["non_nodule_fp"].append(
                (f"{series_uid}_fp_{ii}", {"z1": z1, "y1": y1, "x1": x1, "z2": z2, "y2": y2, "x2": x2})
            )
    return all_preds_dict


def _generate_master_df(vessel_summary_df, all_abn_dicts_per_series_id, num_workers=1):
    # Simple for loop
    master_data = []
    for sid in tqdm(all_abn_dicts_per_series_id.keys(), desc="Generating Master DF"):
        all_abn_dict = all_abn_dicts_per_series_id[sid]
        lung_bbox = torch.load(
            os.path.join(HIPAS_ROOT, f"detection_outputs/preds_raw_sid_wise/{sid}.pth"), weights_only=False
        )["lung_box"]
        lung_bbox = _convert_bbox_from_tensor(lung_bbox)
        safetensor_path = os.path.join(HIPAS_ROOT, f"cts/{sid}.safetensors")
        spacing = {
            "x": load_file(safetensor_path)["spacing_x"].item(),
            "y": load_file(safetensor_path)["spacing_y"].item(),
            "z": load_file(safetensor_path)["spacing_z"].item(),
        }
        for annot_id, bbox in all_abn_dict["pulmonary_vessel"]:
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
                    "split": "train",
                    "dataset_name": "hipas_vessel_fps",
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

    vessel_summary_df = pd.read_csv(os.path.join(HIPAS_ROOT, "detection_outputs/vessel_fp_summary.csv"))

    all_abn_dicts_per_series_id = _generate_all_abn_dicts_per_series_id(vessel_summary_df)

    master_df = _generate_master_df(vessel_summary_df, all_abn_dicts_per_series_id)
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

    master_df_path = os.path.join("hipas_vessel_fp_master.parquet")

    master_df_to_save.to_parquet(master_df_path, index=False, engine="pyarrow")
    print(f"Master dataframe saved to {master_df_path}")
