"""Read byte ranges and tar members from a remote tar file without downloading all of it.

The GrandTour tars on Hugging Face are plain tar files, so an HTTP Range request for the first N bytes yields the first
frames, and walking the 512-byte tar headers locates any member (for example one zarr chunk) inside a 1.4 GB file.
"""
import tarfile
import time
import urllib.request


def fetch(url, start, end, retries=3):
    """Bytes start..end (inclusive) of the file at url."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:
                    raise OSError(f"server ignored the Range header (HTTP {r.status})")
                return r.read()
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def download(url, start, end, dest, step=16 << 20):
    """Stream bytes start..end of url into the file dest, in steps; returns the number of bytes written."""
    n = 0
    with open(dest, "wb") as f:
        for a in range(start, end + 1, step):
            chunk = fetch(url, a, min(a + step, end + 1) - 1)
            f.write(chunk)
            n += len(chunk)
    return n


def find_members(url, wanted):
    """Locate tar members by walking the headers: {name: (data_offset, size)} for every name in `wanted`.

    Reads one 512-byte header per member and skips over the data, so it transfers a few tens of kilobytes.
    """
    wanted, found, off = set(wanted), {}, 0
    while len(found) < len(wanted):
        block = fetch(url, off, off + 511)
        if len(block) < 512 or not block.strip(b"\0"):
            break
        info = tarfile.TarInfo.frombuf(block, tarfile.ENCODING, "surrogateescape")
        if info.name in wanted:
            found[info.name] = (off + 512, info.size)
        off += 512 + ((info.size + 511) // 512) * 512
    missing = wanted - set(found)
    if missing:
        raise FileNotFoundError(f"not found in {url}: {sorted(missing)}")
    return found
