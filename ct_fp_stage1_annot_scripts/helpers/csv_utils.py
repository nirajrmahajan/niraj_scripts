import ast
import os

import loguru
import numpy as np
import pandas as pd
import SimpleITK as sitk
from omegaconf import ListConfig
from safetensors.torch import load_file

FP_REDBRICKS2CSV_MAP = {
    "nodule": "nodule",
    "fibrosis": "fibrosis",
    "apical_pleural_thickening": "apical_pleural_thickening",
    "costochondral_junction_calcification": "costochondral_junction_calcification",
    "osteophyte": "osteophyte",
    "pulmonary_vessel": "pulmonary_vessel",
    "GGO": "ggo",
    "consolidation": "consolidation",
    "pleural_pathology": "pleural_pathology",
    "fissure": "fissure",
    "pleural_effusion": "pleural_effusion",
    "intrapulmonary_lymph_node": "intrapulmonary_lymph_node",
    "atelectasis": "atelectasis",
    "reticulation": "reticulation",
    "other": "other_fp",
    "non_nodule_fp": "non_nodule_fp",
}

ALL_ABN_KEYS = [
    "nodule",
    "fibrosis",
    "apical_pleural_thickening",
    "costochondral_junction_calcification",
    "osteophyte",
    "pulmonary_vessel",
    "ggo",
    "consolidation",
    "pleural_pathology",
    "fissure",
    "pleural_effusion",
    "intrapulmonary_lymph_node",
    "atelectasis",
    "reticulation",
    "other_fp",
    "non_nodule_fp",
]
REQUIRED_COLUMNS = [
    "series_uid",
    "annot_id",
    "safetensor_path",
    "abn_bbox",
    "lung_bbox",
    "all_abn_bboxes_in_sid",
    "spacing",
    "split",
    "dataset_name",
] + ALL_ABN_KEYS


def _validate_bbox_dict(bbox_dict: dict):
    assert isinstance(bbox_dict, dict), f"Invalid value type for bbox_dict: {type(bbox_dict)}. Must be a dict."
    if "x1" in bbox_dict.keys():
        assert set(bbox_dict.keys()) == set(["x1", "y1", "z1", "x2", "y2", "z2"])
    elif "xc" in bbox_dict.keys():
        assert set(bbox_dict.keys()) == set(["xc", "yc", "zc", "w", "h", "d"])
    else:
        raise ValueError(f"Invalid bbox format: {bbox_dict}")


def _eval_if_str(s):
    if isinstance(s, str):
        return ast.literal_eval(s)
    return s


def validate_final_df(df):

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    assert not missing, f"Missing required columns: {missing}"

    df["abn_bbox"].apply(_validate_bbox_dict)
    df["lung_bbox"].apply(_validate_bbox_dict)

    def _validate_abn_dict(abn_dict: dict):
        assert isinstance(abn_dict, dict), f"Invalid value type for abn_dict: {type(abn_dict)}. Must be a dict."
        for key in abn_dict.keys():
            assert key in REQUIRED_COLUMNS, f"Invalid key: {key}. Must be one of {REQUIRED_COLUMNS}"
            assert isinstance(
                abn_dict[key], (list, ListConfig)
            ), f"Invalid value type for {key}: {type(abn_dict[key])}. Must be a list."
            for entry in abn_dict[key]:
                assert isinstance(
                    entry, (list, tuple, ListConfig)
                ), f"Invalid value type for entry: {type(entry)}. Must be a tuple."
                assert len(entry) == 2, f"Invalid length for entry: {len(entry)}. Must be 2."
                assert isinstance(entry[0], str), f"Invalid value type for entry[0]: {type(entry[0])}. Must be a str."
                _validate_bbox_dict(entry[1])

    # Apply _validate_abn_dict on all rows of df['all_abn_bboxes_in_sid'] without a for loop
    df["all_abn_bboxes_in_sid"].apply(_validate_abn_dict)

    def _validate_spacing_dict(spacing_dict: dict):
        assert isinstance(
            spacing_dict, dict
        ), f"Invalid value type for spacing_dict: {type(spacing_dict)}. Must be a dict."
        assert set(spacing_dict.keys()) == set(
            ["x", "y", "z"]
        ), f"Invalid keys for spacing_dict: {spacing_dict.keys()}. Must be one of {'x', 'y', 'z'}"
        assert isinstance(
            spacing_dict["x"], float
        ), f"Invalid value type for spacing_dict['x']: {type(spacing_dict['x'])}. Must be a float."
        assert isinstance(
            spacing_dict["y"], float
        ), f"Invalid value type for spacing_dict['y']: {type(spacing_dict['y'])}. Must be a float."
        assert isinstance(
            spacing_dict["z"], float
        ), f"Invalid value type for spacing_dict['z']: {type(spacing_dict['z'])}. Must be a float."

    df["spacing"].apply(_validate_spacing_dict)

    def _validate_safetensor_path(safetensor_path: str):
        assert os.path.exists(safetensor_path), f"Path {safetensor_path} does not exist."
        data = load_file(safetensor_path)
        assert "ct_data" in data, f"Key 'ct_data' not found in safetensor file {safetensor_path}"

    df["safetensor_path"].apply(_validate_safetensor_path)


def _get_lung_bbox(path):
    assert os.path.exists(path), f"Path {path} does not exist."
    image = sitk.ReadImage(path)
    image_array = sitk.GetArrayFromImage(image)
    assert set(np.unique(image_array).tolist()).issubset(
        set([0, 1, 2])
    ), f"Mask must be 0,1. Got {image_array.unique().tolist()}"
    return _get_strict_bbox_from_mask3d(image_array > 0)


def _get_strict_bbox_from_mask3d(mask):
    # mask is DxHxW
    # return dict with keys {"xc", "yc", "zc", "w", "h", "d"}
    # if mask has all zeros, loguru log that mask is empty and return bbox of whole mask dimensions
    # assert that mask must be 0,1

    # Convert to numpy array if needed
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)

    # Assert mask must be 0,1
    unique_vals = np.unique(mask)
    assert set(unique_vals.tolist()).issubset(set([0, 1])), f"Mask must be 0,1. Got {unique_vals.tolist()}"

    # Get mask dimensions
    d, h, w = mask.shape

    # Check if mask is all zeros
    if np.sum(mask) == 0:
        loguru.logger.warning("Mask is empty (all zeros), returning bbox of whole mask dimensions")
        return {"xc": w / 2.0, "yc": h / 2.0, "zc": d / 2.0, "w": w, "h": h, "d": d}

    # Find bounding box of non-zero values
    # Get indices where mask is non-zero
    nonzero_indices = np.nonzero(mask)

    # Get min and max coordinates for each dimension
    z_min, z_max = np.min(nonzero_indices[0]), np.max(nonzero_indices[0])
    y_min, y_max = np.min(nonzero_indices[1]), np.max(nonzero_indices[1])
    x_min, x_max = np.min(nonzero_indices[2]), np.max(nonzero_indices[2])

    # Calculate center and dimensions
    xc = (x_min + x_max) / 2.0
    yc = (y_min + y_max) / 2.0
    zc = (z_min + z_max) / 2.0

    # Dimensions are inclusive, so add 1
    bbox_w = x_max - x_min + 1
    bbox_h = y_max - y_min + 1
    bbox_d = z_max - z_min + 1

    return {
        "xc": float(xc),
        "yc": float(yc),
        "zc": float(zc),
        "w": float(bbox_w),
        "h": float(bbox_h),
        "d": float(bbox_d),
    }


def _add_data_split_column(df: pd.DataFrame):
    """Add a train/val split while roughly matching abnormality distributions."""
    if len(df) == 0:
        df["split"] = []
        return df

    missing_cols = [col for col in ALL_ABN_KEYS if col not in df.columns]
    assert not missing_cols, f"Missing abnormality columns: {missing_cols}"
    assert "series_uid" in df.columns, "Missing 'series_uid' column."

    def _to_binary(value):
        if pd.isna(value):
            return 0
        if isinstance(value, (list, tuple, set)):
            return int(len(value) > 0)
        if isinstance(value, (bool, np.bool_)):
            return int(value)
        if isinstance(value, (int, np.integer)):
            return int(value > 0)
        if isinstance(value, (float, np.floating)):
            return int(value > 0)
        if isinstance(value, str):
            stripped = value.strip().lower()
            if stripped in {"", "0", "false", "no", "none"}:
                return 0
            if stripped in {"1", "true", "yes"}:
                return 1
            try:
                return int(float(stripped) > 0)
            except ValueError:
                return 1
        return int(bool(value))

    abn_df = df[ALL_ABN_KEYS].applymap(_to_binary)
    total_samples = len(df)
    total_counts = abn_df.sum(axis=0).to_numpy(dtype=float)
    overall_ratio = total_counts / total_samples

    train_target = min(total_samples, max(0, int(round(0.8 * total_samples))))
    if total_samples > 1 and train_target == total_samples:
        train_target -= 1
    val_target = total_samples - train_target

    train_counts = np.zeros(len(ALL_ABN_KEYS), dtype=float)
    val_counts = np.zeros(len(ALL_ABN_KEYS), dtype=float)
    train_size = 0
    val_size = 0

    abn_matrix = abn_df.to_numpy(dtype=float)
    assignments = np.empty(total_samples, dtype=object)
    rng = np.random.default_rng(42)

    series_uids = df["series_uid"].tolist()
    uid_to_indices = {}
    for idx, uid in enumerate(series_uids):
        uid_to_indices.setdefault(uid, []).append(idx)

    group_entries = []
    for indices in uid_to_indices.values():
        idx_array = np.array(indices, dtype=int)
        group_vec = abn_matrix[idx_array].sum(axis=0)
        group_entries.append(
            {
                "indices": idx_array,
                "counts": group_vec,
                "size": len(indices),
            }
        )

    group_order = rng.permutation(len(group_entries))

    def _score(current_counts, current_size, candidate_vec, candidate_size):
        new_counts = current_counts + candidate_vec
        new_size = current_size + candidate_size
        ratios = new_counts / new_size
        return np.abs(ratios - overall_ratio).sum()

    for group_pos in group_order:
        group_entry = group_entries[group_pos]
        row_vec = group_entry["counts"]
        row_size = group_entry["size"]
        if train_size >= train_target:
            choice = "valid_test"
        elif val_size >= val_target:
            choice = "train"
        else:
            train_score = _score(train_counts, train_size, row_vec, row_size)
            val_score = _score(val_counts, val_size, row_vec, row_size)
            if np.isclose(train_score, val_score):
                choice = "train" if train_size <= val_size else "valid_test"
            else:
                choice = "train" if train_score < val_score else "valid_test"

        if choice == "train":
            train_counts += row_vec
            train_size += row_size
        else:
            val_counts += row_vec
            val_size += row_size
        assignments[group_entry["indices"]] = choice

    df["split"] = assignments

    train_mask = df["split"] == "train"
    val_mask = df["split"] == "valid_test"
    overall_dist = (abn_df.sum(axis=0) / total_samples).to_dict()
    train_den = max(train_mask.sum(), 1)
    val_den = max(val_mask.sum(), 1)
    train_dist = (abn_df[train_mask].sum(axis=0) / train_den).to_dict()
    val_dist = (abn_df[val_mask].sum(axis=0) / val_den).to_dict()

    print(f"Total samples: {total_samples}, Train: {train_mask.sum()}, Val: {val_mask.sum()}")
    for key in ALL_ABN_KEYS:
        print(
            f"{key}: train={train_dist.get(key, 0.0):.4f}, "
            f"val={val_dist.get(key, 0.0):.4f}, "
            f"overall={overall_dist.get(key, 0.0):.4f}"
        )

    split_per_uid = df.groupby("series_uid")["split"].nunique()
    invalid_uids = split_per_uid[split_per_uid > 1].index.tolist()
    assert not invalid_uids, f"Series split mismatch for UIDs: {invalid_uids}"

    return df
