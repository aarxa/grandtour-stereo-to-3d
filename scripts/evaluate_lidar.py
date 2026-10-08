#!/usr/bin/env python3
"""Reproduce the numbers in the README: stereo depth against the Hesai LiDAR, with the ZED SDK depth for context.

Run from the repository root after scripts/download_data.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from stereo3d import metrics, stereo as S
from stereo3d.lidar import scan_for

FRAMES = list(range(0, 372, 3))                  # the LiDAR slice covers the first ~25 s, camera frames 0..380
ZED_FRAMES = [i for i in FRAMES if i < 148]      # the downloaded ZED depth slice has frames 0..148


def main():
    usable = [i for i in FRAMES if scan_for(i) is not None]
    print(f"{len(usable)} of {len(FRAMES)} sampled frames have a LiDAR scan within 20 ms (only those are used)\n")

    print("1. Which calibration? Left-referenced stereo, all visible LiDAR returns at 0.5-10 m")
    for flip, rect, label in [(False, None, "parallel cameras, lens undistortion only"),
                              (True, None, "parallel cameras, principal point rotated 180 deg"),
                              (True, "R", "stereo rectification + principal point rotated 180 deg"),
                              (False, "R", "stereo rectification from the dataset extrinsics")]:
        S.configure(flip180=flip, rect=rect)
        zl, zs, _ = metrics.stereo_pairs(FRAMES, "left")
        print(f"   {label:<56} {metrics.summary(zl, zs)}")

    S.configure(flip180=False, rect="R")
    print("\n2. Final calibration, by distance")
    for ref in ("left", "right"):
        zl, zs, hit = metrics.stereo_pairs(FRAMES, ref)
        print(f"   {ref}-referenced ({100 * hit:.0f}% of visible LiDAR returns, incl. beyond 10 m, have a stereo value): "
              f"{metrics.summary(zl, zs)}")
        print("\n".join(metrics.band_lines(zl, zs)))

    zl, zd = metrics.zed_pairs(ZED_FRAMES)
    print(f"\n3. ZED SDK depth map vs LiDAR (its own coverage, frames 0-148): {metrics.summary(zl, zd)}")
    print("\n".join(metrics.band_lines(zl, zd)))

    zl, zs, zz = metrics.like_for_like(ZED_FRAMES)
    both = np.isfinite(zs) & np.isfinite(zz)
    print(f"\n4. Like for like: the {both.sum()} visible LiDAR returns (frames 0-148) where both methods have a value")
    print(f"   coverage of the {len(zl)} visible returns at 0.5-10 m: stereo {100 * np.isfinite(zs).mean():.0f}%, "
          f"ZED SDK {100 * np.isfinite(zz).mean():.0f}%, both {100 * both.mean():.0f}%")
    print(f"   stereo  {metrics.summary(zl[both], zs[both])}")
    print(f"   ZED SDK {metrics.summary(zl[both], zz[both])}")


if __name__ == "__main__":
    main()
