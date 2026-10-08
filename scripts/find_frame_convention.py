#!/usr/bin/env python3
"""Find, by brute force, how the metadata transforms map the LiDAR into the camera frames.

The metadata yaml files give box_base -> sensor transforms, but it is easy to read them the wrong way round, and the
camera axes could also be permuted. This script tries every combination against the ZED depth map:

    2 readings of the transform  x  2 cameras  x  24 axis permutations  = 96 hypotheses

For each one it transforms a LiDAR scan into the camera frame, projects it with the ZED depth intrinsics and checks how
many returns agree with the ZED depth within 15 %. The right combination stands out clearly (about 86 % against under
30 % for everything else), and fixes the convention used in stereo3d/lidar.py.
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from stereo3d.dataset import depth_intrinsics, read_zed_depth, sensor_tf
from stereo3d.lidar import scan_for

FRAMES = [20, 50, 80, 110]
READINGS = {"yaml is the sensor pose in box_base": lambda R, t: (R, t),
            "yaml maps box_base into the sensor":   lambda R, t: (R.T, -R.T @ t)}


def axis_rotations():
    """The 24 proper rotations that permute and flip the coordinate axes."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for row, (col, s) in enumerate(zip(perm, signs)):
                M[row, col] = s
            if np.linalg.det(M) > 0:
                out.append(M)
    return out


def project(p_cam, depth, K):
    """(lidar_z, depth_z) for the LiDAR returns that land on a valid pixel of the depth map."""
    z = p_cam[:, 2]
    front = z > 0.3
    u = np.round(K[0, 0] * p_cam[front, 0] / z[front] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * p_cam[front, 1] / z[front] + K[1, 2]).astype(int)
    h, w = depth.shape
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    zd = depth[v[inside], u[inside]]
    zl = z[front][inside]
    good = np.isfinite(zd) & (zd > 0.3) & (zd < 15)
    return zl[good], zd[good]


def main():
    K = depth_intrinsics()
    data = [(scan_for(i, max_dt=0.06), read_zed_depth(i)) for i in FRAMES]
    results = []
    for reading, convert in READINGS.items():
        R_l, t_l = convert(*sensor_tf("hesai_points"))
        for cam in ("left", "right"):
            R_c, t_c = convert(*sensor_tf(f"zed2i_{cam}_images"))
            for M in axis_rotations():
                zl, zd = [], []
                for p, depth in data:
                    p_box = p @ R_l.T + t_l
                    a, b = project(((p_box - t_c) @ R_c) @ M.T, depth, K)
                    zl.append(a); zd.append(b)
                zl, zd = np.concatenate(zl), np.concatenate(zd)
                if len(zl) >= 3000:
                    rel = np.abs(zl - zd) / zd
                    results.append((np.mean(rel < 0.15), np.median(rel), len(zl), reading, cam, M))
    results.sort(key=lambda r: -r[0])
    print(f"{len(results)} of 96 hypotheses put at least 3000 LiDAR returns inside the image. Best first:\n")
    for inliers, med, n, reading, cam, M in results[:6]:
        print(f"within 15%: {100 * inliers:3.0f}%  median rel err {100 * med:5.1f}%  n={n:>6}  {reading:<38} camera={cam:<5} "
              f"axes={'identity' if np.allclose(M, np.eye(3)) else M.astype(int).tolist()}")


if __name__ == "__main__":
    main()
