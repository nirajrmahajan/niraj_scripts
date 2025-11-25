import glob
import os
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np


def blend_images_with_ratios(
    image_paths: Union[List[str], Tuple[str, str, str]],
    ratios: Union[List[float], Tuple[float, float, float]] = (1.0, 1.0, 1.0),
    output_path: str = None,
    normalize: bool = True,
    clip_output: bool = True,
    rotate_180: bool = False,
    output_dtype: np.dtype = np.uint8,
) -> np.ndarray:
    """
    Blend 3 images with specified ratios and optionally save the result.

    Parameters
    ----------
    image_paths : list or tuple of str
        Paths to 3 images to blend
    ratios : list or tuple of float
        Blend ratios for each image (will be normalized to sum to 1.0)
        Default: (1.0, 1.0, 1.0) - equal weight
    output_path : str, optional
        Path to save the blended image. If None, image is not saved.
    normalize : bool
        If True, normalize ratios so they sum to 1.0
    clip_output : bool
        If True, clip output values to valid range for the output dtype
    output_dtype : np.dtype
        Data type of output image (default: np.uint8 for 8-bit images)

    Returns
    -------
    np.ndarray
        Blended image as numpy array

    Example
    -------
    >>> lung_xr = "/path/to/lung.png"
    >>> soft_tissue_xr = "/path/to/nonlung.png"
    >>> bone_xr = "/path/to/bone.png"
    >>>
    >>> blended = blend_images_with_ratios(
    ...     [lung_xr, soft_tissue_xr, bone_xr],
    ...     ratios=(1.75, 3.5, 1.0),
    ...     output_path="/path/to/output.png"
    ... )
    """

    # Validate inputs
    if len(image_paths) != 3:
        raise ValueError(f"Expected 3 images, got {len(image_paths)}")

    if len(ratios) != 3:
        raise ValueError(f"Expected 3 ratios, got {len(ratios)}")

    # Normalize ratios if requested
    ratios = np.array(ratios, dtype=np.float32)
    if normalize:
        ratios = ratios / np.sum(ratios)

    # Load images
    images = []
    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        images.append(img.astype(np.float32))
        print(f"Loaded image {i+1}: {img_path}, shape: {img.shape}")

    # Verify all images have the same shape
    shape = images[0].shape
    for i, img in enumerate(images[1:], 1):
        if img.shape != shape:
            raise ValueError(
                f"Image {i+1} has shape {img.shape}, expected {shape}. " "All images must have the same dimensions."
            )

    # Blend images with specified ratios
    blended = images[0] * ratios[0] + images[1] * ratios[1] + images[2] * ratios[2]

    # Clip to valid range if requested
    if clip_output:
        if output_dtype == np.uint8:
            blended = np.clip(blended, 0, 255)
        elif output_dtype == np.uint16:
            blended = np.clip(blended, 0, 65535)
        elif output_dtype == np.float32:
            blended = np.clip(blended, 0, 1)

    # Convert to output dtype
    blended = blended.astype(output_dtype)

    if rotate_180:
        blended = cv2.rotate(blended, cv2.ROTATE_180)
        print("✓ Image rotated by 180°")

    # Save if output path provided
    # plt.imshow(blended, cmap='gray')
    # plt.axis('off')
    # plt.show()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(output_path, blended)
        if success:
            print(f"✓ Blended image saved to: {output_path}")
        else:
            print(f"✗ Failed to save blended image to: {output_path}")

    return blended


def combine_all_folder_projections(xray_projections_base_path, save_base_path):
    xray_projections_full = [
        os.path.join(xray_projections_base_path, x) for x in sorted(os.listdir(xray_projections_base_path))
    ]

    for pid in xray_projections_full:

        png_xray = sorted(glob.glob(os.path.join(pid, "*.png")))
        # ap_9kv_xray = [f for f in png_xray if "AP_9kV" in f]
        # ap_11kv_xray = [f for f in png_xray if "AP_11kV" in f]
        ap_13kv_xray = [f for f in png_xray if "AP_13kV" in f]
        ap_17kv_xray = [f for f in png_xray if "AP_17kV" in f]
        ap_25kv_xray = [f for f in png_xray if "AP_25kV" in f]
        # ap_35kv_xray = [f for f in png_xray if "AP_35kV" in f]
        # ap_50kv_xray = [f for f in png_xray if "AP_50kV" in f]
        # ap_100kv_xray = [f for f in png_xray if "AP_100kV" in f]
        # pa_9kv_xray = [f for f in png_xray if "PA_9kV" in f]
        # pa_13kv_xray = [f for f in png_xray if "PA_13kV" in f]
        # pa_50kv_xray = [f for f in png_xray if "PA_50kV" in f]

        all_files = {
            # "ap_9kv_xray": ap_9kv_xray,
            # "ap_11kv_xray": ap_11kv_xray,
            "ap_13kv_xray": ap_13kv_xray,
            "ap_17kv_xray": ap_17kv_xray,
            "ap_25kv_xray": ap_25kv_xray,
            # "ap_35kv_xray": ap_35kv_xray,
            # "ap_50kv_xray": ap_50kv_xray,
            # "ap_100kv_xray": ap_100kv_xray,
        }
        for k, v in all_files.items():
            try:
                lung_xr = v[2]
                soft_tissue_xr = v[3]
                bone_xr = v[0]
            except IndexError:
                print(f"⚠️  Skipping {os.path.basename(pid)}: Not enough files for {k} (found {len(v)})")
                continue

            blended = blend_images_with_ratios(
                [lung_xr, soft_tissue_xr, bone_xr],
                ratios=(1.5, 4.0, 1.0),
                output_path=f"{save_base_path}/{os.path.basename(pid)}/{k}.png",
            )

        # if len(ap_9kv_files) < 3:
        #     print(f"⚠️  Skipping {os.path.basename(pid)}: Found only {len(ap_9kv_files)} AP_9kV files (need 3)")
        #     continue


if __name__ == "__main__":
    xray_projections_base = "/home/users/niraj.mahajan/projects/scripts/ct2xr/penrad_x-ray_projections"
    save_base_path = "/home/users/niraj.mahajan/projects/scripts/ct2xr/penrad_x-ray_projections_combined"

    combine_all_folder_projections(xray_projections_base, save_base_path)
