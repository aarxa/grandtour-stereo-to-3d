"""Depth-error metrics: stereo depth (and the ZED SDK depth) against the Hesai LiDAR."""
import cv2
import numpy as np

from . import stereo as S
from .dataset import depth_intrinsics, read_pair, read_zed_depth
from .lidar import sample_pairs, scan_for, to_camera

BINS = [(0.5, 2), (2, 4), (4, 6), (6, 10)]       # LiDAR distance bands in metres


def _kernel():
    return int(30 * S.SCALE), int(10 * S.SCALE)


def stereo_pairs(frames, ref="left"):
    """(lidar_z, stereo_z, fraction of visible LiDAR returns that have a stereo value) over `frames`.

    Only frames with a LiDAR scan within 20 ms are used, so robot motion does not blur the comparison.
    """
    zl, zs, n_visible = [], [], 0
    for i in frames:
        p = scan_for(i)
        if p is None:
            continue
        _, d, K = S.disparity_right(*read_pair(i)) if ref == "right" else S.disparity(*read_pair(i))
        pc = to_camera(p, ref) @ S.R_RECT[1 if ref == "right" else 0].T
        a, b, n = sample_pairs(pc, S.depth_from_disparity(d, K), K, _kernel())
        zl.append(a); zs.append(b); n_visible += n
    zl, zs = np.concatenate(zl), np.concatenate(zs)
    return zl, zs, len(zl) / max(n_visible, 1)


def zed_pairs(frames):
    """(lidar_z, zed_sdk_z) over `frames` that have a LiDAR scan within 20 ms (ZED depth is registered to the right camera)."""
    K = depth_intrinsics()
    zl, zd = [], []
    for i in frames:
        p = scan_for(i)
        if p is None:
            continue
        a, b, _ = sample_pairs(to_camera(p, "right"), read_zed_depth(i), K, (30, 10))
        zl.append(a); zd.append(b)
    return np.concatenate(zl), np.concatenate(zd)


def like_for_like(frames):
    """Stereo and ZED SDK depth at the same visible LiDAR returns: (lidar_z, stereo_z, zed_z), NaN where a method has no value."""
    K_zed = depth_intrinsics()
    ZL, ZS, ZZ = [], [], []
    for i in frames:
        p = scan_for(i)
        if p is None:
            continue
        _, d, K = S.disparity_right(*read_pair(i))
        zs = S.depth_from_disparity(d, K)
        zed = read_zed_depth(i)
        p_right = to_camera(p, "right")
        pc = p_right @ S.R_RECT[1].T
        z = pc[:, 2]
        ok = (z > 0.5) & (z < 10)
        pc, p_right, z = pc[ok], p_right[ok], z[ok]
        u = np.round(K[0, 0] * pc[:, 0] / z + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * pc[:, 1] / z + K[1, 2]).astype(int)
        h, w = zs.shape
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        pc, p_right, z, u, v = pc[inside], p_right[inside], z[inside], u[inside], v[inside]
        nearest = np.full((h, w), np.inf, np.float32)
        np.minimum.at(nearest, (v, u), z.astype(np.float32))
        visible = z <= 1.10 * cv2.erode(nearest, np.ones(_kernel(), np.uint8))[v, u]
        p_right, z, u, v = p_right[visible], z[visible], u[visible], v[visible]
        # the ZED depth map has its own pinhole model, so project the same 3D points with those intrinsics
        uz = np.round(K_zed[0, 0] * p_right[:, 0] / p_right[:, 2] + K_zed[0, 2]).astype(int)
        vz = np.round(K_zed[1, 1] * p_right[:, 1] / p_right[:, 2] + K_zed[1, 2]).astype(int)
        inside_z = (uz >= 0) & (uz < zed.shape[1]) & (vz >= 0) & (vz < zed.shape[0])
        zz = np.full(len(z), np.nan, np.float32)
        zz[inside_z] = zed[vz[inside_z], uz[inside_z]]
        ZL.append(z); ZS.append(zs[v, u]); ZZ.append(zz)
    return np.concatenate(ZL), np.concatenate(ZS), np.concatenate(ZZ)


def error_stats(zl, zo):
    """Median absolute error (cm), median relative error (%), share within 10 %, median signed bias (%)."""
    rel = (zo - zl) / zl
    return dict(n=len(zl), err_cm=100 * np.median(np.abs(zo - zl)), rel=100 * np.median(np.abs(rel)),
                within10=100 * np.mean(np.abs(rel) < 0.1), bias=100 * np.median(rel))


def summary(zl, zo):
    s = error_stats(zl, zo)
    return (f"n={s['n']:>6} | median |err| {s['err_cm']:5.1f} cm | median |rel| {s['rel']:4.1f}% | "
            f"within 10%: {s['within10']:3.0f}% | bias {s['bias']:+5.1f}%")


def band_lines(zl, zo, bins=BINS, min_points=50):
    return [f"    LiDAR {a:>3}-{b:<3} m: {summary(zl[m], zo[m])}"
            for a, b in bins for m in [(zl >= a) & (zl < b)] if m.sum() > min_points]
