"""Stereo rectification, SGBM disparity and depth for the GrandTour ZED 2i pair.

Call ``configure`` to choose the camera model (it runs once with the best settings at import):

    rect="R"   rectify with the stereo extrinsics from the mission metadata (default, most accurate)
    rect=None  assume parallel cameras and only undo lens distortion (the naive baseline)
    flip180    rotate the principal point by 180 degrees; a wrong theory kept for the comparison table
"""
import os

import cv2
import numpy as np

from .dataset import IMAGE_H, IMAGE_W, camera_info, sensor_tf

SCALE = float(os.environ.get("STEREO_SCALE", 0.5))     # 0.5 = work at 960x540; full resolution is 4x slower, not better
SIZE = (int(IMAGE_W * SCALE), int(IMAGE_H * SCALE))
NUM_DISPARITIES = int(256 * SCALE)                      # SGBM cannot match the leftmost columns, so those are invalid

sgbm = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=NUM_DISPARITIES, blockSize=5,
    P1=8 * 3 * 25, P2=32 * 3 * 25, uniquenessRatio=10,
    speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)

# set by configure()
KV = None          # intrinsics of the virtual rectified camera (at SIZE)
BASELINE = None    # metres
MAPS = None        # remap tables (left, right)
R_RECT = None      # rotation applied to the (left, right) camera frames by the rectification


def _camera(side, flip180):
    K, d = camera_info(side)
    if flip180:
        K[0, 2], K[1, 2] = IMAGE_W - 1 - K[0, 2], IMAGE_H - 1 - K[1, 2]
        d[2:4] *= -1
    return K, d


def configure(flip180=False, rect="R"):
    global KV, BASELINE, MAPS, R_RECT
    S = np.diag([SCALE, SCALE, 1.0])
    (KL, DL), (KR, DR) = _camera("left", flip180), _camera("right", flip180)
    KL, KR = S @ KL, S @ KR
    (Rl, tl), (Rr, tr) = sensor_tf("zed2i_left_images"), sensor_tf("zed2i_right_images")
    R = Rr @ Rl.T                  # left optical frame -> right optical frame
    T = tr - R @ tl
    if rect is None:
        KV, BASELINE, R_RECT = KL.copy(), float(np.linalg.norm(T)), (np.eye(3), np.eye(3))
        MAPS = (cv2.initUndistortRectifyMap(KL, DL, None, KV, SIZE, cv2.CV_32FC1),
                cv2.initUndistortRectifyMap(KR, DR, None, KV, SIZE, cv2.CV_32FC1))
        return
    R1, R2, P1, P2, *_ = cv2.stereoRectify(KL, DL.reshape(-1, 1), KR, DR.reshape(-1, 1), SIZE,
                                           np.ascontiguousarray(R), T.reshape(3, 1),
                                           flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    KV, BASELINE, R_RECT = P1[:3, :3].copy(), float(-P2[0, 3] / P2[0, 0]), (R1, R2)
    MAPS = (cv2.initUndistortRectifyMap(KL, DL, R1, KV, SIZE, cv2.CV_32FC1),
            cv2.initUndistortRectifyMap(KR, DR, R2, KV, SIZE, cv2.CV_32FC1))


configure()


def undistort_pair(left, right):
    """Rectified (left, right) images at SIZE plus the intrinsics of the rectified virtual camera."""
    ls = cv2.resize(left, SIZE, interpolation=cv2.INTER_AREA)
    rs = cv2.resize(right, SIZE, interpolation=cv2.INTER_AREA)
    return cv2.remap(ls, *MAPS[0], cv2.INTER_LINEAR), cv2.remap(rs, *MAPS[1], cv2.INTER_LINEAR), KV


def sgbm_disparity(ref, other):
    gray = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    d = sgbm.compute(gray(ref), gray(other)).astype(np.float32) / 16.0
    d[d < 1] = np.nan
    return d


def disparity(left, right):
    """(rectified left image, disparity in the left pixel grid, intrinsics)."""
    lu, ru, K = undistort_pair(left, right)
    return lu, sgbm_disparity(lu, ru), K


def disparity_right(left, right):
    """(rectified right image, disparity in the right pixel grid, intrinsics); mirror trick: swap the roles and flip."""
    lu, ru, K = undistort_pair(left, right)
    return ru, sgbm_disparity(ru[:, ::-1], lu[:, ::-1])[:, ::-1], K


def depth_from_disparity(d, K):
    """Metric depth along the optical axis: Z = f * B / d."""
    return K[0, 0] * BASELINE / d


def to_points(img, d, K, max_depth=12.0):
    """Back-project a disparity map to a coloured point cloud. Returns (xyz, rgb in 0..1, depth map)."""
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    v, u = np.indices(d.shape)
    z = depth_from_disparity(d, K)
    ok = np.isfinite(z) & (z < max_depth)
    x = (u - cx) * z / f
    y = (v - cy) * z / f
    return np.stack([x, y, z], -1)[ok], img[..., ::-1][ok] / 255.0, z
