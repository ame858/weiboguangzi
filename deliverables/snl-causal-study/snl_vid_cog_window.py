"""Read a small planar UInt16 ICEYE video window by validated HTTP ranges."""
import argparse
import hashlib
import io
import struct
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from snl_inputs import DEFAULT_OUTPUT, ROOT, read_json, sha_file, write_json

sys.path.insert(0, str(ROOT))
from videosar.data import run_directory


LIMA = "ICEYE_6MC5KR_20250526T033241Z_4770346_X31_SLEDF"
EXTERNAL = DEFAULT_OUTPUT / "external-sar"
PREVIEW_BANDS = (0, 2, 4)
TILE_COLUMN_ROW = (2, 6)


def read_range(url, offset, length, *, etag=None, tries=3):
    end = offset + length - 1
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{end}"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(request, timeout=35) as response:
                span = response.headers["Content-Range"]
                received_etag = response.headers["ETag"]
                if response.status != 206 or not span or not span.startswith(f"bytes {offset}-{end}/"):
                    raise ValueError(f"Unexpected response for the requested byte range {offset}–{end}")
                if etag and received_etag != etag:
                    raise ValueError("The public COG changed during the tile read")
                payload = response.read(length)
                if len(payload) != length:
                    raise OSError("Incomplete public COG byte range")
            return payload, int(span.split("/")[1]), received_etag
        except OSError as error:
            if attempt + 1 == tries:
                raise RuntimeError(f"Could not read COG byte range {offset}–{end}") from error


def decode_lzw(payload, width, height, endian, predictor):
    """Decode one compressed planar tile using Pillow's native TIFF decoder."""
    marker = b"II" if endian == "<" else b"MM"
    entries = [(256, 4, width), (257, 4, height), (258, 3, 16), (259, 3, 5), (262, 3, 1),
               (273, 4, 0), (277, 3, 1), (278, 4, height), (279, 4, len(payload)), (284, 3, 1), (317, 3, predictor)]
    start = 8 + 2 + 12 * len(entries) + 4
    table = b"".join(struct.pack(endian + "HHI", tag, kind, 1) +
                     struct.pack(endian + ("H" if kind == 3 else "I"), start if tag == 273 else value).ljust(4, b"\0")
                     for tag, kind, value in entries)
    tiff = marker + struct.pack(endian + "HIH", 42, 8, len(entries)) + table + struct.pack(endian + "I", 0) + payload
    with Image.open(io.BytesIO(tiff)) as image:
        return np.asarray(image, dtype=np.uint16).copy()


def layout(page):
    if page.dtype != np.uint16 or page.samplesperpixel != 25 or int(page.planarconfig) != 2 or int(page.compression) != 5:
        raise ValueError("This range reader requires a 25-band, planar-separate UInt16 LZW video page")
    width, height = page.imagewidth, page.imagelength
    tile_width, tile_height = page.tilewidth, page.tilelength
    nx, ny = (width + tile_width - 1) // tile_width, (height + tile_height - 1) // tile_height
    offsets, lengths = page.dataoffsets, page.databytecounts
    if len(offsets) != 25 * nx * ny or len(lengths) != len(offsets):
        raise ValueError("Video tile index is incomplete")
    return dict(width=width, height=height, tile_width=tile_width, tile_height=tile_height, tiles_x=nx, tiles_y=ny,
                offsets=offsets, lengths=lengths, predictor=int(page.predictor))


def fetch(output, item):
    metadata = read_json(item)
    url = metadata["assets"]["vid-cog"]["href"].replace("s3.amazonaws.com", "s3.us-west-2.amazonaws.com")
    header, length, etag = read_range(url, 0, 65536)
    with tifffile.TiffFile(io.BytesIO(header)) as directory:
        endian = directory.byteorder
        full, reduced = layout(directory.pages[0]), layout(directory.pages[2])
    width, height = full["width"], full["height"]
    x, y = TILE_COLUMN_ROW
    xr, yr = reduced["tiles_x"], reduced["tiles_y"]
    if not (0 <= x < full["tiles_x"] and 0 <= y < full["tiles_y"]):
        raise ValueError("Requested full-resolution tile is outside the video")
    requests = [("preview", reduced, i, tx, ty) for i in PREVIEW_BANDS for ty in range(yr) for tx in range(xr)]
    requests.extend(("native", full, i, x, y) for i in range(25))
    cache_key = hashlib.sha256(f"{url}:{etag}".encode()).hexdigest()[:16]
    cache = output.parent / "vid-cog-range-cache" / cache_key
    cache.mkdir(parents=True, exist_ok=True)

    def cached_range(offset, count):
        path = cache / f"{offset}-{count}.bin"
        digest = path.with_suffix(".sha256")
        if path.is_file() and digest.is_file():
            data = path.read_bytes()
            if len(data) == count and hashlib.sha256(data).hexdigest() == digest.read_text().strip():
                return data
        try:
            data, file_size, _ = read_range(url, offset, count, etag=etag, tries=1)
        except RuntimeError:
            sizes = [(start, min(65536, offset + count - start)) for start in range(offset, offset + count, 65536)]
            with ThreadPoolExecutor(max_workers=4) as retry:
                options = retry.map(lambda pair: read_range(url, *pair, etag=etag), sizes)
                chunks = list(options)
            if any(size != length for _, size, _ in chunks):
                raise ValueError("The COG file size changed during the tile read")
            data, file_size = b"".join(payload for payload, _, _ in chunks), length
        if len(data) != count or file_size != length:
            raise ValueError("Incomplete tile or changed COG file size")
        path.write_bytes(data)
        digest.write_text(hashlib.sha256(data).hexdigest(), encoding="ascii")
        return data

    def tile(spec):
        kind, grid, band, tx, ty = spec
        i = (band * grid["tiles_y"] + ty) * grid["tiles_x"] + tx
        offset, count = grid["offsets"][i], grid["lengths"][i]
        if not (0 < count <= 2_000_000 and 0 <= offset <= length - count):
            raise ValueError("A selected tile is empty, oversized, or outside the source")
        data = cached_range(offset, count)
        return kind, band, tx, ty, count, decode_lzw(data, grid["tile_width"], grid["tile_height"], endian, grid["predictor"])

    settings = dict(stac_id=metadata["id"], source_url=url, source_bytes=length, source_etag=etag,
                    full_shape_yx=[height, width], tile_xy=[x, y], origin_xy=[x * full["tile_width"], y * full["tile_height"]],
                    preview_shape_yx=[reduced["height"], reduced["width"]], preview_bands=list(PREVIEW_BANDS))
    with run_directory(output, "snl-vid-cog-window", settings) as root:
        native = np.empty((25, full["tile_height"], full["tile_width"]), dtype=np.uint16)
        preview = np.empty((len(PREVIEW_BANDS), yr * reduced["tile_height"], xr * reduced["tile_width"]), dtype=np.uint16)
        transferred = len(header)
        with ThreadPoolExecutor(max_workers=12) as pool:
            futures = [pool.submit(tile, request) for request in requests]
            for n, future in enumerate(as_completed(futures), 1):
                kind, band, tx, ty, count, pixels = future.result()
                transferred += count
                if kind == "native":
                    native[band] = pixels
                else:
                    top, left = ty * reduced["tile_height"], tx * reduced["tile_width"]
                    preview[PREVIEW_BANDS.index(band), top:top + pixels.shape[0], left:left + pixels.shape[1]] = pixels
                if n % 5 == 0 or n == len(futures):
                    print(f"Read and decoded {n}/{len(futures)} public COG tiles", flush=True)
        np.save(root / "native-tile.npy", native)
        np.save(root / "overview.npy", preview[:, :reduced["height"], :reduced["width"]])
        hashes = {name: sha_file(root / name) for name in ("native-tile.npy", "overview.npy")}
        write_json(root / "receipt.json", settings | dict(requested_asset_bytes=transferred, array_sha256=hashes))
        print(f"Decoded all {len(native)} native tile bands from {transferred:,} selected bytes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", type=Path, default=EXTERNAL / "iceye-open" / f"{LIMA}.json")
    parser.add_argument("--output", type=Path, default=EXTERNAL / "lima-vid-cog-window")
    args = parser.parse_args()
    fetch(args.output.resolve(), args.item.resolve())
