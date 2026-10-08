#!/usr/bin/env python3
"""Download the slice of one GrandTour mission that this project uses (about 0.5 GB, not the 2 TB dataset).

Source: https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset  (ETH Zurich, Robotic Systems Lab)

What is fetched, for mission 2024-11-14-13-45-37 by default:
  metadata/   camera calibration and sensor extrinsics (a few KB)
  images/     the first --zed-mb of the left/right ZED 2i image tars and --depth-mb of the ZED depth tar
              (a tar is sequential, so a byte range from the start is a valid archive of the first frames)
  zarr/       camera timestamps, plus the first 256 Hesai LiDAR scans (about 25 s). The 1.4 GB LiDAR tar is
              indexed by walking its tar headers, then only the three arrays needed are fetched (about 110 MB)

Files land in data/ (override with GRANDTOUR_DATA). Existing files are kept unless --force is given.
"""
import argparse
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stereo3d import remote
from stereo3d.dataset import DATA

BASE = "https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset/resolve/main"
METADATA = ["zed2i_left_caminfo", "zed2i_right_caminfo", "zed2i_left_images", "zed2i_right_images",
            "zed2i_depth_image", "zed2i_depth_caminfo", "hesai_points"]
LIDAR_MEMBERS = ["points/.zarray", "points/0.0.0", "timestamp/.zarray", "timestamp/0", "valid/.zarray", "valid/0.0"]


def extract(tar_path, out_dir):
    """Extract regular files from a possibly truncated tar, skipping the cut-off last member. Returns the file count."""
    out_dir, n = Path(out_dir).resolve(), 0
    with tarfile.open(tar_path) as tar:
        try:
            for member in tar:
                if not member.isreg():
                    continue
                dest = (out_dir / member.name).resolve()
                if out_dir not in dest.parents:
                    raise ValueError(f"refusing to extract outside {out_dir}: {member.name}")
                data = tar.extractfile(member).read()
                if len(data) != member.size:
                    break
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                n += 1
        except tarfile.ReadError:
            pass                                    # truncated archive: everything before the cut is extracted
    return n


def partial_tar(url, megabytes, out_dir, keep):
    """Download the first `megabytes` MiB of a tar and extract the complete files in it."""
    tmp = DATA / "_partial" / Path(url).name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    remote.download(url, 0, int(megabytes * 1024 * 1024) - 1, tmp)
    n = extract(tmp, out_dir)
    if not keep:
        tmp.unlink()
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", default="2024-11-14-13-45-37")
    ap.add_argument("--zed-mb", type=float, default=150, help="MiB of each left/right image tar (about 3.4 frames per MiB)")
    ap.add_argument("--depth-mb", type=float, default=100, help="MiB of the ZED depth tar (about 1.5 frames per MiB)")
    ap.add_argument("--skip-lidar", action="store_true", help="skip the 110 MB LiDAR slice")
    ap.add_argument("--keep-tars", action="store_true", help="keep the partial tar files")
    ap.add_argument("--force", action="store_true", help="download again even if the files exist")
    args = ap.parse_args()
    root = f"{BASE}/{args.mission}"

    print(f"metadata -> {DATA / 'metadata'}")
    (DATA / "metadata").mkdir(parents=True, exist_ok=True)
    for name in METADATA:
        dest = DATA / "metadata" / f"{name}.yaml"
        if args.force or not dest.exists():
            dest.write_bytes(remote.fetch(f"{root}/metadata/{name}.yaml", 0, 1 << 20))   # a few hundred bytes each

    for topic, mb, first in [("zed2i_left_images", args.zed_mb, "000000.jpeg"),
                             ("zed2i_right_images", args.zed_mb, "000000.jpeg"),
                             ("zed2i_depth_image", args.depth_mb, "000000.png")]:
        if not args.force and (DATA / "images" / topic / first).exists():
            print(f"images/{topic}: already there")
            continue
        print(f"images/{topic}: first {mb:g} MiB ...", flush=True)
        n = partial_tar(f"{root}/images/{topic}.tar", mb, DATA / "images", args.keep_tars)
        print(f"   {n} frames")

    topic = "zed2i_right_images"                       # timestamps; the depth map shares the right camera's
    if args.force or not (DATA / "zarr" / topic / "timestamp" / "0").exists():
        print(f"zarr/{topic}: timestamps", flush=True)
        partial_tar(f"{root}/data/{topic}.tar", 16, DATA / "zarr", args.keep_tars)   # the whole tar is 2 MB

    try:
        (DATA / "_partial").rmdir()                    # empty unless --keep-tars
    except OSError:
        pass
    if args.skip_lidar:
        return
    lidar_dir = DATA / "zarr" / "hesai_points"
    if not args.force and all((lidar_dir / m).exists() for m in LIDAR_MEMBERS):
        print("zarr/hesai_points: already there")
        return
    url = f"{root}/data/hesai_points.tar"
    print("zarr/hesai_points: indexing the tar headers (about a minute) ...", flush=True)
    found = remote.find_members(url, [f"hesai_points/{m}" for m in LIDAR_MEMBERS])
    for name, (offset, size) in found.items():
        print(f"   {name} ({size / 1e6:.1f} MB)", flush=True)
        dest = DATA / "zarr" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        remote.download(url, offset, offset + size - 1, dest)
    print("done")


if __name__ == "__main__":
    main()
