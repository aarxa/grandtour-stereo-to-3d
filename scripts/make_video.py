#!/usr/bin/env python3
"""Render the demo video: the stereo demo (camera, depth, orbiting 3D cloud) followed by the LiDAR check.

Every number shown in the video is computed here, not typed in. Run from the repository root after
scripts/download_data.py. Writes out/linkedin.mp4 plus two stills (out/thumbnail.png, out/validation.png).
"""
import functools
import os
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import imageio_ffmpeg
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from stereo3d import metrics, stereo as S, viz
from stereo3d.dataset import read_pair
from stereo3d.lidar import sample_pairs, scan_for, to_camera

assert S.SCALE == 0.5, "the video layout is designed for the default STEREO_SCALE=0.5"

W, H, FPS = 1920, 1080, 15
BG, PANEL, TEXT, MUTED, ACCENT = (26, 20, 18), (38, 30, 27), (248, 244, 242), (178, 163, 154), (197, 209, 79)   # BGR
DEMO = (290, 490)       # frames with no people or number plates in view
VAL = (300, 372)        # frames that have a LiDAR scan
CROP = S.NUM_DISPARITIES                         # leftmost columns SGBM cannot match
LW = S.SIZE[0] - CROP                            # width of the camera / depth panels (832)
RW = W - LW                                      # width of the 3D panel (1088)
LUT = (plt.get_cmap("coolwarm")(np.linspace(0, 1, 256))[:, :3][:, ::-1] * 255).astype(np.uint8)   # BGR
AVENIR = "/System/Library/Fonts/Avenir Next.ttc"
STATS = {}


# ---------------------------------------------------------------- text helpers
@functools.lru_cache(None)
def font(size, weight="regular"):
    if os.path.exists(AVENIR) and not os.environ.get("NO_AVENIR"):
        return ImageFont.truetype(AVENIR, size, index={"bold": 0, "demi": 2, "medium": 5, "regular": 7}[weight])
    name = "DejaVuSans-Bold.ttf" if weight in ("bold", "demi") else "DejaVuSans.ttf"     # ships with matplotlib
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name), size)


def put(img, xy, text, size, color=TEXT, weight="regular", anchor="la"):
    pil = Image.fromarray(np.ascontiguousarray(img[..., ::-1]))
    ImageDraw.Draw(pil).text(xy, text, font=font(size, weight), fill=color[::-1], anchor=anchor)
    img[:] = np.asarray(pil)[..., ::-1]


@functools.lru_cache(None)
def strip(text, width):
    img = np.full((52, width, 3), BG, np.uint8)
    put(img, (18, 26), text, 27, TEXT, "demi", "lm")
    return img


def labelled(panel, text, bottom=False):
    out = panel.copy()
    sl = slice(-52, None) if bottom else slice(0, 52)
    out[sl] = (0.82 * strip(text, panel.shape[1]) + 0.18 * out[sl]).astype(np.uint8)
    return out


def init_worker(stats):
    global STATS
    STATS = stats
    cv2.setNumThreads(1)


# ---------------------------------------------------------------- scene 1: stereo demo
def demo_frame(args):
    i, yaw = args
    lu, d, K = S.disparity(*read_pair(i))
    dv = np.nan_to_num(d, nan=0)
    med = cv2.medianBlur(dv.astype(np.float32), 5)                  # drop isolated mismatches from the 3D view only
    d_clean = np.where(np.abs(dv - med) <= np.maximum(1.0, 0.1 * dv), d, np.nan)
    xyz, rgb, _ = S.to_points(lu, d_clean, K, max_depth=10.0)
    colour = cv2.applyColorMap(np.clip(dv / 90 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colour[dv == 0] = 0
    left = np.vstack([labelled(lu[:, CROP:], "Left camera, ANYmal ZED 2i"),
                      labelled(colour[:, CROP:], "Depth from stereo  (warm = near)")])
    cloud = labelled(viz.render_cloud(xyz, rgb, yaw, size=(RW, H)), "3D point cloud rebuilt from the stereo pair")
    cloud = labelled(cloud, f"{100 * S.BASELINE:.0f} cm baseline  ·  SGBM disparity  ·  depth = f × B ÷ disparity", bottom=True)
    return np.hstack([left, cloud])


# ---------------------------------------------------------------- scene 2: LiDAR check
PX, PY, PW, PH = 40, 150, 1200, 675              # camera image panel


@functools.lru_cache(None)
def validation_background():
    st = STATS
    c = np.full((H, W, 3), BG, np.uint8)
    put(c, (40, 38), "Checked against LiDAR ground truth", 54, TEXT, "bold")
    put(c, (40, 104), "Each dot is one Hesai LiDAR return, coloured by the stereo depth error at that point.", 27, MUTED)
    cv2.rectangle(c, (PX - 3, PY - 3), (PX + PW + 2, PY + PH + 2), PANEL, 3)
    x0 = 1320
    put(c, (x0, 150), f"{st['rel']:.1f}%", 128, ACCENT, "bold")
    put(c, (x0, 296), "median depth error", 34, TEXT, "medium")
    put(c, (x0, 344), f"{st['w10']:.0f}% of points within 10%", 32, MUTED)
    put(c, (x0, 440), "Median error by distance", 30, TEXT, "demi")
    longest = max(cm for _, cm in st["bins"])
    for k, (label, cm) in enumerate(st["bins"]):
        y = 506 + k * 74
        put(c, (x0, y), label, 27, MUTED)
        bar = max(8, int(300 * cm / longest))
        cv2.rectangle(c, (x0 + 150, y + 3), (x0 + 150 + bar, y + 35), ACCENT, -1)
        put(c, (x0 + 150 + bar + 14, y), f"{cm:.0f} cm", 27, TEXT, "medium")
    lx, ly, lw = PX, PY + PH + 38, 720              # colour legend
    c[ly:ly + 22, lx:lx + lw] = np.repeat(LUT[np.linspace(0, 255, lw).astype(int)][None], 22, axis=0)
    put(c, (lx, ly + 32), "-15%   stereo reads too close", 24, MUTED)
    put(c, (lx + lw, ly + 32), "stereo reads too far   +15%", 24, MUTED, anchor="ra")
    put(c, (PX, 1000), f"{st['n']:,} LiDAR points over {st['frames']} frames, 0.5 to 10 m. "
                       "Dataset calibration, stereo rectification, classical SGBM.", 24, MUTED)
    return c


def validation_frame(i):
    lu, d, K = S.disparity(*read_pair(i))
    pc = to_camera(scan_for(i, max_dt=0.06), "left") @ S.R_RECT[0].T
    zl, zs, _, pixels = sample_pairs(pc, S.depth_from_disparity(d, K), K, (15, 5), with_pixels=True)
    rel = (zs - zl) / zl
    img = (0.7 * cv2.resize(lu, (PW, PH), interpolation=cv2.INTER_LINEAR)).astype(np.uint8)
    colours = LUT[np.clip((rel + 0.15) / 0.30 * 255, 0, 255).astype(int)]
    sx, sy = PW / lu.shape[1], PH / lu.shape[0]
    for (u, v), colour in zip(pixels, colours):
        cv2.circle(img, (int(u * sx), int(v * sy)), 3, colour.tolist(), -1, cv2.LINE_AA)
    out = validation_background().copy()
    out[PY:PY + PH, PX:PX + PW] = img
    return out


# ---------------------------------------------------------------- assembly
def compute_stats():
    frames = list(range(0, 372, 3))
    zl, zs, _ = metrics.stereo_pairs(frames, "left")
    overall = metrics.error_stats(zl, zs)
    bins = [(f"{a:g}–{b:g} m", metrics.error_stats(zl[m], zs[m])["err_cm"])
            for a, b in metrics.BINS for m in [(zl >= a) & (zl < b)]]
    stats = dict(rel=overall["rel"], w10=overall["within10"], n=overall["n"], bins=bins,
                 frames=sum(scan_for(i) is not None for i in frames))
    print({k: (round(float(v), 2) if not isinstance(v, (list, int)) else v) for k, v in stats.items()})
    return stats


def main(out="out/linkedin.mp4"):
    os.makedirs("out", exist_ok=True)
    stats = compute_stats()
    ids = list(range(*DEMO))
    yaws = [np.radians(26) * np.sin(2 * np.pi * k / len(ids) * 1.25) for k in range(len(ids))]
    with Pool(initializer=init_worker, initargs=(stats,)) as pool:
        demo = list(pool.imap(demo_frame, zip(ids, yaws), chunksize=4))
        check = list(pool.imap(validation_frame, range(*VAL), chunksize=2))
    cv2.imwrite("out/thumbnail.png", demo[len(demo) // 2])
    cv2.imwrite("out/validation.png", check[len(check) // 2])

    writer = imageio_ffmpeg.write_frames(out, (W, H), fps=FPS, codec="libx264", quality=6, macro_block_size=1)
    writer.send(None)

    def emit(frames, fade_in=6, fade_out=0):
        for k, f in enumerate(frames):
            a = min(1.0, (k + 1) / fade_in)
            if fade_out:
                a = min(a, (len(frames) - k) / fade_out)
            img = f if a >= 1 else (f * a).astype(np.uint8)
            writer.send(np.ascontiguousarray(img[..., ::-1]))

    emit(demo)
    emit(check + [check[-1]] * int(2 * FPS), fade_out=8)
    writer.close()
    print(f"wrote {out}: {(len(demo) + len(check) + 2 * FPS) / FPS:.1f} s")


if __name__ == "__main__":
    main()
