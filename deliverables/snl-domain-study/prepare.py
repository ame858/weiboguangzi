"""Prepare paired SAR training windows for the SNL domain ablation."""
import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs/snl-eubank-check"
DEFAULT_OUTPUT = ROOT / "runs/snl-domain-ablation"
SIM = ROOT / "runs/contrast-study/dataset"
sys.path[:0] = [str(ROOT), str(SOURCE)]

from factorial import SIZE, features, rasterize
from videosar.simulation import paired_shadow_video

SIZES = (1.0, 1.35, 1.60, 1.85, 2.10)
RETENTIONS = (1.0, 0.30, 0.45, 0.60, 0.75)
GROUPS = ("A", "B", "C", "D")
FRAMES = (3, 6, 9, 12, 15, 18)
TEST_BLOCKS = {
    "332-374": tuple(range(332, 375, 6)),
    "560-602": tuple(range(560, 603, 6)),
    "830-872": tuple(range(830, 873, 6)),
}
COUNT = 69


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(array):
    return sha256(array.tobytes()).hexdigest()


def simulate(task):
    cache, scene_path, size, retentions, code_hash = task
    scene_meta = scene_path / "metadata.json"
    definition = read_json(scene_meta)
    source = definition["effective_config"]
    config = source | {
        key: source[key] * size for key in ("object_width_m", "object_depth_m", "object_height_m")
    }
    key = sha256(repr((scene_meta.read_bytes(), size, retentions, code_hash)).encode()).hexdigest()[:20]
    path = cache / f"{scene_path.name}-{key}.npz"
    if path.exists():
        return (scene_path.name, size), path

    frames, visible = [], []
    reference_hash = geometry_hash = None
    noise_level = None
    for _, (observed, mask, meta, geometry, reference) in paired_shadow_video(
        config, retentions, 25, definition["static_dark"]
    ):
        current_ref = digest(reference)
        current_geom = digest(geometry)
        current_noise = meta["receiver_noise_power"]
        if reference_hash is None:
            reference_hash, geometry_hash, noise_level = current_ref, current_geom, current_noise
        if (reference_hash, geometry_hash, noise_level) != (current_ref, current_geom, current_noise):
            raise ValueError(f"{scene_path.name}: reference, geometry, or noise changed at fixed size")
        if abs(meta["snr_measured_db"] - 25) > 0.1 or not np.isfinite(observed).all():
            raise ValueError(f"{scene_path.name}: invalid observations or SNR")
        frames.append(observed.astype(np.float32))
        visible.append(mask)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, frames=np.stack(frames), visible=np.stack(visible),
                        retentions=np.asarray(retentions), reference_hash=reference_hash,
                        geometry_hash=geometry_hash, receiver_noise_power=noise_level)
    temporary.replace(path)
    return (scene_path.name, size), path


def generate(pool, output, cache, scenes, sizes, retentions, code_hash):
    tasks = [(cache, SIM / scene, size, retentions, code_hash) for scene in scenes for size in sizes]
    archives = {}
    for count, (key, path) in enumerate(pool.map(simulate, tasks), 1):
        archives[key] = path
        if count % 12 == 0 or count == len(tasks):
            print(f"{output}: {count}/{len(tasks)} paired apertures ready", flush=True)
    return archives


def bounds(mask):
    pixels = np.argwhere(mask)
    if not len(pixels):
        return None
    y0, x0 = pixels.min(axis=0)
    y1, x1 = pixels.max(axis=0) + 1
    return np.array([x0, y0, x1, y1])


def contrast(log_image, target):
    local = binary_dilation(target, iterations=6) & ~binary_dilation(target, iterations=2)
    return float(np.median(log_image[local]) - np.median(log_image[target]))


def calibration(archives, scenes, target_area, target_contrast):
    rows = []
    for size in SIZES:
        areas, darkness, retained = [], [[] for _ in RETENTIONS], [[] for _ in RETENTIONS]
        for scene in scenes:
            with np.load(archives[scene, size]) as data:
                frames, visible = data["frames"], data["visible"]
            for n in FRAMES:
                box = bounds(visible[0, n])
                if box is None:
                    continue
                areas.append(int(np.prod(box[2:] - box[:2])))
                target = rasterize(box[None], visible.shape[-2:]) > 0
                for i in range(len(RETENTIONS)):
                    log_image = features(frames[i, n], None)[0]
                    darkness[i].append(contrast(log_image, target))
                    retained[i].append(bool(np.any(visible[0, n] & visible[i, n])))
        for i, retention in enumerate(RETENTIONS):
            rows.append(dict(size=size, retention=retention, positive_frames=len(areas),
                             box_area_median=float(np.median(areas)),
                             contrast_median=float(np.median(darkness[i])),
                             frames_with_remaining_visible_pixels=sum(retained[i])))
    size = min((r for r in rows if r["size"] > 1 and r["retention"] == 1),
               key=lambda r: (abs(r["box_area_median"] - target_area), r["size"]))["size"]
    retention = min(RETENTIONS[1:], key=lambda r: (
        sum(abs(row["contrast_median"] - target_contrast) for row in rows
            if row["size"] in (1.0, size) and row["retention"] == r),
        -r,
    ))
    return rows, size, retention


def make_windows(archives, positive_scenes, empty_scenes, size, retention):
    rng = np.random.default_rng(717)
    originals, enlarged, archives_opened = {}, {}, {}
    for scene in (*positive_scenes, *empty_scenes):
        first = archives[scene, 1.0]
        second = archives[scene, size] if scene in positive_scenes else first
        with np.load(first) as data:
            originals[scene] = dict(data)
        with np.load(second) as data:
            enlarged[scene] = dict(data)
        base, large = originals[scene], enlarged[scene]
        for key in ("reference_hash", "receiver_noise_power"):
            if base[key] != large[key]:
                raise ValueError(f"{scene}: size change altered the paired background or receiver noise")
        archives_opened[scene] = (first.name, second.name)

    candidates = []
    missing = 0
    for scene in positive_scenes:
        for n in FRAMES:
            base = bounds(originals[scene]["visible"][0, n])
            large = bounds(enlarged[scene]["visible"][0, n])
            if base is None or large is None:
                missing += 1
                continue
            candidates.append((scene, n, base, large))
    positive = [candidates[i] for i in rng.choice(len(candidates), COUNT, replace=False)]
    empty = [(scene, n, None, None) for scene in empty_scenes for n in FRAMES]
    negative = [empty[i] for i in rng.choice(len(empty), COUNT, replace=False)]
    x, y = {g: [] for g in GROUPS}, {g: [] for g in GROUPS}
    windows, distributions = [], []
    for scene, n, small, large in (*positive, *negative):
        base, big = originals[scene], enlarged[scene]
        h, w = base["frames"].shape[-2:]
        if small is None:
            if base["visible"].any():
                raise ValueError("Empty controls contain visible target pixels")
            left, top = rng.integers([0, 0], [w-SIZE+1, h-SIZE+1])
        else:
            lower, upper = np.minimum(small[:2], large[:2]), np.maximum(small[2:], large[2:])
            minimum = np.maximum(0, upper - SIZE)
            maximum = np.minimum(lower, (w - SIZE, h - SIZE))
            if np.any(minimum > maximum):
                raise ValueError(f"{scene} frame {n}: paired target does not fit a shared window")
            jitter = rng.integers(-9, 10, size=2)
            left, top = np.clip((lower + upper - SIZE)//2 + jitter, minimum, maximum)
        windows.append(dict(scene=scene, frame=n, left=int(left), top=int(top),
                            positive=small is not None, archive_by_size=archives_opened[scene]))
        for group, variant, box, level in zip(
            GROUPS, (base, big, base, big), (small, large, small, large), (1.0, 1.0, retention, retention),
        ):
            i = list(variant["retentions"]).index(level) if small is not None else 0
            observed = variant["frames"][i]
            inputs = features(observed[n], observed[n-1])[:, top:top+SIZE, left:left+SIZE]
            target = np.zeros((SIZE, SIZE), np.float32)
            if box is not None:
                shifted = box - [left, top, left, top]
                target = rasterize(shifted[None], (SIZE, SIZE))
                visible = variant["visible"][0, n] & variant["visible"][i, n]
                distributions.append(dict(group=group, scene=scene, frame=n, area=int(target.sum()),
                                          local_contrast=contrast(inputs[0], target > 0),
                                          remaining_visible_pixels=int(visible.sum())))
            x[group].append(inputs)
            y[group].append(target)
    arrays = {f"{kind}_{group}": np.stack(mapping[group])
              for kind, mapping in (("x", x), ("y", y)) for group in GROUPS}
    for a, b in (("A", "C"), ("B", "D")):
        if not np.array_equal(arrays[f"y_{a}"], arrays[f"y_{b}"]):
            raise ValueError("Contrast groups have different training labels")
    for g in GROUPS:
        if arrays[f"y_{g}"][:COUNT].sum() == 0 or arrays[f"y_{g}"][COUNT:].any():
            raise ValueError("Invalid positive or negative training window")
        if not np.array_equal(arrays[f"x_A"][COUNT:], arrays[f"x_{g}"][COUNT:]):
            raise ValueError("Negative controls differ across groups")
    return arrays, windows, distributions, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    out = args.output.resolve()
    cache = out / "echo-cache"
    cache.mkdir(parents=True, exist_ok=True)
    with (SIM / "manifest.csv").open(newline="") as file:
        rows = [r for r in csv.DictReader(file) if r["split"] == "train"]
    small = sorted(row["path"] for row in rows if row["group"] == "small")
    normal = sorted(row["path"] for row in rows if row["group"] == "normal")
    empty = sorted(row["path"] for row in rows if row["group"] == "empty")
    pilot = small[:8] + normal[:4]
    positive = sorted(small + normal)
    target = read_json(SOURCE / "factorial-training-distribution.json")["data"]["real"]
    area = target["selected_box_area_q25_median_q75_pixel2"][1]
    darkness = target["normalized_log_darkness_ring_minus_box_median_q25_median_q75"][1]
    code_hash = sha256((ROOT / "videosar/simulation.py").read_bytes()).hexdigest()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        stage = generate(pool, "Calibration", cache, pilot, SIZES, RETENTIONS, code_hash)
        grid, size, retention = calibration(stage, pilot, area, darkness)
        save_csv(out / "calibration.csv", grid)
        print(f"Frozen calibration: size {size:.2f}, attenuation retention {retention:.2f}", flush=True)
        files = generate(pool, "Paired training", cache, sorted(set(positive) - set(pilot)),
                         (1.0, size), (1.0, retention), code_hash)
        files |= {key: value for key, value in stage.items() if key[1] in (1.0, size)}
        files |= generate(pool, "Negative controls", cache, empty, (1.0,), (1.0,), code_hash)
    arrays, windows, distributions, missing = make_windows(files, positive, empty, size, retention)
    save_csv(out / "training-distribution.csv", distributions)
    save_json(out / "paired-windows.json", windows)
    np.savez_compressed(out / "training.npz", **arrays)
    save_json(out / "simulation-protocol.json", {
        "source_manifest_sha256": sha256((SIM / "manifest.csv").read_bytes()).hexdigest(),
        "simulation_sha256": code_hash, "calibration_scenes": pilot, "sampled_frame_indices": FRAMES,
        "training_window_size": SIZE, "size_grid": SIZES, "contrast_retention_grid": RETENTIONS,
        "calibration_target_area": area, "calibration_target_contrast": darkness,
        "size_scale": size, "attenuation_retention": retention,
        "uncropped_training_frames_without_box_in_at_least_one_size": missing,
        "training_npz_sha256": sha256((out / "training.npz").read_bytes()).hexdigest(),
        "test_annotation_blocks": TEST_BLOCKS,
    })
    print(f"Saved {COUNT} shared positive and {COUNT} shared negative windows for each group.", flush=True)


if __name__ == "__main__":
    main()
