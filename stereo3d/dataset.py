"""Paths, calibration files and frame loading for the downloaded slice of a GrandTour mission.

Everything lives under ``data/`` (override with the GRANDTOUR_DATA environment variable). The layout mirrors the
Hugging Face repository, see scripts/download_data.py:

    data/metadata/   calibration and extrinsics (yaml)
    data/images/     zed2i_left_images/, zed2i_right_images/, zed2i_depth_image/  (numbered frames)
    data/zarr/       timestamps and the Hesai LiDAR slice (zarr arrays)
"""
import functools
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

DATA = Path(os.environ.get("GRANDTOUR_DATA", "data"))
IMAGE_W, IMAGE_H = 1920, 1080


def metadata(name):
    with open(DATA / "metadata" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def camera_info(side):
    """Intrinsic matrix K (3x3) and distortion coefficients of the ZED 2i 'left' or 'right' camera, as shipped."""
    c = metadata(f"zed2i_{side}_caminfo")["camera_info"]
    return np.array(c["K"], float).reshape(3, 3), np.array(c["D"], float)


def depth_intrinsics():
    """Pinhole intrinsics of the ZED SDK depth map (registered to the right camera)."""
    return np.array(metadata("zed2i_depth_caminfo")["camera_info"]["K"], float).reshape(3, 3)


@functools.lru_cache(None)
def sensor_tf(name):
    """Rotation matrix and translation from a metadata file such as ``hesai_points`` or ``zed2i_left_images``.

    The file says base_frame_id: box_base, child_frame_id: <sensor>. Registering the LiDAR to the ZED depth map
    (scripts/find_frame_convention.py) shows that these numbers map box_base coordinates *into* the sensor frame:

        p_sensor = R @ p_box + t

    and that the camera frames are already optical frames (x right, y down, z forward).
    """
    t = metadata(name)["transform"]
    q = t["rotation"]
    w, x, y, z = q["w"], q["x"], q["y"], q["z"]
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return R, np.array([t["translation"][k] for k in "xyz"])


def _read(path, flags):
    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(f"{path} is missing; run scripts/download_data.py (or the frame is beyond the downloaded slice)")
    return img


def read_pair(i):
    """Left and right ZED 2i images of frame i (BGR, 1920x1080)."""
    return (_read(DATA / "images" / "zed2i_left_images" / f"{i:06d}.jpeg", cv2.IMREAD_COLOR),
            _read(DATA / "images" / "zed2i_right_images" / f"{i:06d}.jpeg", cv2.IMREAD_COLOR))


def read_zed_depth(i):
    """ZED SDK depth map of frame i in metres (NaN where invalid), registered to the right camera."""
    d = _read(DATA / "images" / "zed2i_depth_image" / f"{i:06d}.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    d[d == 0] = np.nan
    return d
