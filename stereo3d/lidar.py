"""Hesai LiDAR scans from the downloaded zarr slice, and LiDAR -> camera projection helpers."""
import functools
import json

import cv2
import numcodecs
import numpy as np

from .dataset import DATA, sensor_tf

SCANS_PER_CHUNK = 256      # chunking of hesai_points/points: (256 scans, 69000 padded points, xyz)
POINTS_PER_SCAN = 69000


def _read_chunk(array_dir, chunk):
    """Decode one chunk of a zarr v2 array (blosc-compressed) into raw bytes."""
    codec = numcodecs.get_codec(json.loads((array_dir / ".zarray").read_text())["compressor"])
    return codec.decode((array_dir / chunk).read_bytes())


@functools.lru_cache(None)
def load_scans():
    """(points[256, 69000, 3], timestamps[256], valid_counts[256]) for the first 25 s of the mission."""
    root = DATA / "zarr" / "hesai_points"
    points = np.frombuffer(_read_chunk(root / "points", "0.0.0"), "<f4").reshape(SCANS_PER_CHUNK, POINTS_PER_SCAN, 3)
    ts = np.frombuffer(_read_chunk(root / "timestamp", "0"), "<f8")[:SCANS_PER_CHUNK]
    valid = np.frombuffer(_read_chunk(root / "valid", "0.0"), "<u4")[:SCANS_PER_CHUNK]
    return points, ts, valid


@functools.lru_cache(None)
def camera_timestamps(topic="zed2i_right_images"):
    """Timestamps (epoch seconds) of every frame of a camera topic; the depth map shares the right camera's."""
    array_dir = DATA / "zarr" / topic / "timestamp"
    n = json.loads((array_dir / ".zarray").read_text())["shape"][0]
    return np.frombuffer(_read_chunk(array_dir, "0"), "<f8")[:n]


def scan_for(i, max_dt=0.02):
    """Valid points of the LiDAR scan closest in time to camera frame i, or None if none is within max_dt seconds."""
    points, ts, valid = load_scans()
    t = camera_timestamps()[i]
    j = int(np.argmin(np.abs(ts - t)))
    if abs(ts[j] - t) > max_dt:
        return None
    p = points[j][:valid[j]]
    return p[np.isfinite(p).all(1) & (np.abs(p).sum(1) > 0)]


def to_camera(points, side):
    """LiDAR points (Hesai frame) -> optical frame of the 'left' or 'right' ZED camera."""
    R_lidar, t_lidar = sensor_tf("hesai_points")
    R_cam, t_cam = sensor_tf(f"zed2i_{side}_images")
    p_box = (points - t_lidar) @ R_lidar          # p_box = R^T (p - t), see dataset.sensor_tf
    return p_box @ R_cam.T + t_cam


def sample_pairs(pc, depth, K, kernel, with_pixels=False):
    """Depth values at the pixels of the LiDAR returns that the camera can actually see.

    pc     LiDAR points in the camera frame of `depth` (N, 3)
    depth  depth map in metres, NaN where invalid
    kernel (rows, cols) size of the neighbourhood used to drop returns hidden behind nearer returns
    Returns (lidar_z, depth_z, n_visible[, pixels]) for visible points with 0.5 < z < 10 m and a valid depth value.
    """
    z = pc[:, 2]
    ok = z > 0.5
    pc, z = pc[ok], z[ok]
    u = K[0, 0] * pc[:, 0] / z + K[0, 2]
    v = K[1, 1] * pc[:, 1] / z + K[1, 2]
    h, w = depth.shape
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    ui, vi, z = ui[inside], vi[inside], z[inside]
    nearest = np.full((h, w), np.inf, np.float32)
    np.minimum.at(nearest, (vi, ui), z.astype(np.float32))
    nearest = cv2.erode(nearest, np.ones(kernel, np.uint8))      # nearest LiDAR return around each pixel
    visible = z <= 1.10 * nearest[vi, ui]                        # drop returns hidden behind nearer ones
    zd, zl = depth[vi[visible], ui[visible]], z[visible]
    good = np.isfinite(zd) & (zl < 10)
    if with_pixels:
        return zl[good], zd[good], int(visible.sum()), np.c_[ui[visible][good], vi[visible][good]]
    return zl[good], zd[good], int(visible.sum())
