from hashlib import sha256

import numpy as np
from PIL import Image

from .config import SIMULATION, merged
from .data import save_sequence, write_json
from .simulation import form_video, simulate
from .visuals import write_csv


def translate(frame, dx, dy):
    pad = int(np.ceil(max(abs(dx), abs(dy)))) + 2
    padded = np.pad(np.asarray(frame, dtype=np.float32), pad, mode="reflect")
    height, width = frame.shape
    return np.asarray(Image.fromarray(padded).transform(
        (width, height), Image.Transform.AFFINE, (1, 0, pad - dx, 0, 1, pad - dy), Image.Resampling.BILINEAR))


def scene_digest(path):
    digest = sha256()
    for name in ("frames.npy", "masks.npy", "geometric_masks.npy", "metadata.json"):
        digest.update((path / name).read_bytes())
    return digest.hexdigest()


def make_proposal_dataset(root, count, seed):
    if count < 100:
        raise ValueError("the proposal profile requires at least 100 independent sequences")
    train_end, validation_end = count * 3 // 5, count * 4 // 5
    splits = (("train", 0, train_end), ("validation", train_end, validation_end), ("test", validation_end, count))
    rows, labels = [], []
    for split, start, stop in splits:
        for index in range(start, stop):
            local = index - start
            held_out = split == "test" and local >= stop - start - 6
            source = "sar" if local % 3 == 1 else "geometry"
            speed_band = (local + local // 3) % 3
            empty = local % 6 == 4
            high_noise = local % 6 in (2, 5)
            jittered = local % 5 in (2, 4)
            background = "roads" if held_out else ("uniform", "texture", "clutter")[(local // 3) % 3]
            scene_id = f"proposal-{index:03d}"
            rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
            speed = rng.uniform(*((0.65, 1.25), (1.45, 2.15), (2.50, 3.55))[speed_band])
            if held_out and speed_band == 2:
                speed += 0.65
            heading = rng.choice([-1, 1])
            velocity_x = float(heading * speed)
            config = merged(SIMULATION, {"scene": "empty" if empty else "geometry", "background": background,
                "frames": 20, "height": 64, "width": 80, "start_x_m": float(-velocity_x * 1.9 + rng.uniform(-0.8, 0.8)),
                "start_y_m": float(rng.uniform(-3.8, -0.4)), "velocity_x_mps": velocity_x,
                "velocity_y_mps": float(rng.uniform(-0.65, 0.65)), "object_height_m": float(rng.uniform(0.8, 2.45)),
                "object_width_m": float(rng.uniform(0.65, 1.6)), "object_depth_m": float(rng.uniform(0.75, 1.6)),
                "shadow_factor": float(rng.uniform(0.045, 0.22)), "noise": float(rng.uniform(0.22, 0.36) if high_noise else rng.uniform(0.045, 0.14)),
                "thermal_noise": float(rng.uniform(0.20, 0.36) if high_noise else rng.uniform(0.055, 0.12)),
                "seed": int(rng.integers(0, 2**32 - 1)), "pulses_per_frame": 61,
                "carrier_hz": float(rng.uniform(34e9, 36e9)), "bandwidth_hz": float(rng.uniform(550e6, 720e6))})
            if source == "sar":
                frames, masks, meta, geometry, _ = form_video(config, normalize_power=False)
            else:
                frames, masks, meta, _ = simulate(config, ground_speckle=True)
                geometry = masks
            if bool(masks.any()) == empty:
                raise ValueError(f"{scene_id}: generated foreground does not match target availability")
            offsets = np.zeros((len(frames), 2))
            if jittered:
                offsets[1:] = rng.uniform(-1.65, 1.65, size=(len(frames) - 1, 2))
                frames = np.stack([translate(frame, dx, dy) for frame, (dx, dy) in zip(frames, offsets)])
            condition = "held_out_background" if held_out else ("empty" if empty else ("high_noise" if high_noise else "standard"))
            row = {"path": scene_id, "scene_id": scene_id, "trajectory_family": scene_id, "split": split,
                   "source": source, "condition": condition, "background": background,
                   "speed_band": None if empty else ("slow", "medium", "fast")[speed_band],
                   "noise_band": "high" if high_noise else "normal", "registration_perturbation": jittered}
            meta.update(row, observed_offsets_xy_px=offsets.tolist(), label_coordinates="fixed_ground_grid")
            save_sequence(root / scene_id, frames, masks, meta, geometry)
            row["sha256"] = scene_digest(root / scene_id)
            rows.append(row)
            labels.append(dict(row, frames=len(frames), visible_frames=int(masks.any(axis=(1, 2)).sum()),
                               foreground_pixels=int(masks.sum()), pixels=int(masks.size)))
    write_csv(root / "labels_by_scene.csv", labels)
    write_json(root / "dataset.json", {"version": 1, "profile": "proposal", "seed": seed, "entries": rows})
