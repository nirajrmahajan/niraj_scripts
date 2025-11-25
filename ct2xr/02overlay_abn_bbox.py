import argparse
import ast
import os
from multiprocessing import Pool

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def _eval_if_str(inp):
    if isinstance(inp, str):
        return ast.literal_eval(inp)
    return inp


def _read_bbox(bbox):
    bbox = _eval_if_str(bbox)
    if "x1" in bbox.keys():
        return bbox
    else:
        x1 = bbox["xc"] - (bbox["w"] / 2)
        y1 = bbox["yc"] - (bbox["h"] / 2)
        z1 = bbox["zc"] - (bbox["d"] / 2)
        x2 = bbox["xc"] + (bbox["w"] / 2)
        y2 = bbox["yc"] + (bbox["h"] / 2)
        z2 = bbox["zc"] + (bbox["d"] / 2)
        return {"x1": x1, "y1": y1, "z1": z1, "x2": x2, "y2": y2, "z2": z2}


def get_abn_dict(series_uid, abn_df):
    all_rows = abn_df.loc[series_uid]
    if isinstance(all_rows, pd.Series):
        all_rows = pd.DataFrame([all_rows])
    abn_dict = {}
    for i in range(len(all_rows)):
        row = all_rows.iloc[i]
        bbox = _read_bbox(_eval_if_str(row["annot"])["bbox"])
        meta = _eval_if_str(row["meta"])
        if meta["nodule_type"] == "true positive":
            abn_dict["nodule"] = abn_dict.get("nodule", []) + [bbox]
        elif meta["nodule_type"] == "false positive":
            abn_dict[meta["reason_for_false_positive"]] = abn_dict.get(meta["reason_for_false_positive"], []) + [bbox]
    return abn_dict


def _flip_bbox(bbox, image):
    return {
        "z1": image.shape[0] - bbox["z2"],
        "y1": bbox["y1"],
        "x1": bbox["x1"],
        "z2": image.shape[0] - bbox["z1"],
        "y2": bbox["y2"],
        "x2": bbox["x2"],
    }


def _load_ct_from_sid(series_uid, voltage=25):
    path = f"./penrad_x-ray_projections_combined/{series_uid}/ap_{voltage}kv_xray.png"
    assert os.path.exists(path), f"Path {path} does not exist"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_flipped = np.flipud(img)
    return img_flipped


def _get_bbox_area(bbox):
    return (bbox["y2"] - bbox["y1"]) * (bbox["z2"] - bbox["z1"])


def process_sid(series_uid, abn_df, voltage=25):
    abn_df = get_abn_dict(series_uid, abn_df)
    image = _load_ct_from_sid(series_uid, voltage)
    assert abn_df, f"No ABN found for {series_uid}"
    for key, bbox_list in abn_df.items():
        abn_image = image.copy()
        max_bbox_area = 0
        for bbox in bbox_list:
            bbox = _flip_bbox(bbox, image)
            bbox_area = _get_bbox_area(bbox)
            if bbox_area > max_bbox_area:
                max_bbox_area = bbox_area
            abn_image = overlay_bbox_on_image(abn_image, bbox, color="red", thickness=2)
        if max_bbox_area > 1000:
            size_str = "big"
        else:
            size_str = "tiny"
        os.makedirs(f"./abn_bboxes_{size_str}/{series_uid}", exist_ok=True)
        cv2.imwrite(f"./abn_bboxes_{size_str}/{series_uid}/ap_{voltage}kv_{key}.png", abn_image)
        if not os.path.exists(f"./abn_bboxes_{size_str}/{series_uid}/ap_{voltage}kv_normal.png"):
            cv2.imwrite(f"./abn_bboxes_{size_str}/{series_uid}/ap_{voltage}kv_normal.png", image)


def overlay_bbox_on_image(image, bbox_dict, color="red", thickness=2):
    """Draw a bounding box on the image and return an RGB array."""
    color_map = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }

    img = np.asarray(image).copy()
    if img.ndim == 2:
        img_rgb = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 3:
        img_rgb = img
    else:
        raise ValueError("Image must be either HxW or HxWx3")

    if isinstance(color, str):
        color_rgb = color_map.get(color.lower())
        if color_rgb is None:
            raise ValueError(f"Unsupported color: {color}")
    else:
        color_rgb = tuple(color)
        if len(color_rgb) != 3:
            raise ValueError("Custom color must be an RGB tuple of length 3")

    try:
        z1, y1, z2, y2 = [int(round(bbox_dict[k])) for k in ("z1", "y1", "z2", "y2")]
    except KeyError as exc:
        raise KeyError("bbox_dict must contain x1, y1, x2, y2") from exc

    img_bgr = np.ascontiguousarray(img_rgb[..., ::-1])
    cv2.rectangle(img_bgr, (z1, y1), (z2, y2), color_rgb[::-1], thickness)
    return img_bgr[..., ::-1].copy()


def _init_pool(abn_df):
    """Initializer to share abn_df across pool workers."""
    global _ABN_DF
    _ABN_DF = abn_df


def _process_sid_with_global(series_uid):
    process_sid(series_uid, _ABN_DF, voltage=25)
    process_sid(series_uid, _ABN_DF, voltage=17)
    process_sid(series_uid, _ABN_DF, voltage=13)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers to use.")
    args = parser.parse_args()

    abn_df = pd.read_csv(
        "/cache/fast_data_nas8/qct_segmentations/annotations/download/qCT_Lung_Cancer_penrad_6_stage_1_0925/annotations.csv"
    ).set_index("series_uid")
    all_sids_saved = os.listdir("./penrad_x-ray_projections_combined")
    # all_sids_saved = ["1.2.826.0.1.3867976.3.10061.66341.20240716163237.2503"]
    abn_df = abn_df[abn_df.index.isin(all_sids_saved)]
    all_sids = list(abn_df.index.unique())
    if args.num_workers == 1:
        for series_uid in tqdm(all_sids, desc="Processing series"):
            process_sid(series_uid, abn_df, voltage=25)
            process_sid(series_uid, abn_df, voltage=17)
            process_sid(series_uid, abn_df, voltage=13)
    else:
        with Pool(processes=args.num_workers, initializer=_init_pool, initargs=(abn_df,)) as pool:
            for _ in tqdm(
                pool.imap_unordered(_process_sid_with_global, all_sids), total=len(all_sids), desc="Processing series"
            ):
                pass
