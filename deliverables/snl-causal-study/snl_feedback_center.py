"""Train a causal detector that consumes its own previous center-confidence map."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from audit_detection import paired_block_f1
from experiment import filter_by_config, load_candidates, quality, select_operation, summarize, warm_ids, with_nms
from snl_bytetrack import REFERENCE
from snl_center import CenterDetector, center_loss, decode_center_map
from snl_detector import TILE, backend, draw_windows, target_in_tile, training_tiles
from snl_inputs import DEFAULT_OUTPUT, PROFILES, feature_cache, load_gray, load_labels, read_json, sha_file, write_csv, write_json
from snl_protocol import passed
from videosar.snl_causal import CausalFeatures, detection_counts


SEED = 1729
STEPS = 3
CHECKPOINT_EPOCHS = (6, 12, 18)
BATCH_SIZE = 10
HISTORY_DROP = 0.20
FIRST_WEIGHT = "fine.0.0.weight"


class HistoryAlignment(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Sequential(nn.Conv2d(4, 16, 5, padding=2), nn.SiLU(),
                                    nn.Conv2d(16, 16, 3, padding=1), nn.SiLU(), nn.Conv2d(16, 2, 3, padding=1))
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)

    def forward(self, frame, history):
        height, width = history.shape[-2:]
        image = F.interpolate(frame[:, :2], size=(height, width), mode="area")
        offset = 5 * self.offset(torch.cat((image, image[:, :1] - image[:, 1:], history), dim=1)).tanh()
        x = (torch.arange(width, device=frame.device, dtype=frame.dtype) + 0.5) * (2 / width) - 1
        y = (torch.arange(height, device=frame.device, dtype=frame.dtype) + 0.5) * (2 / height) - 1
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1) + offset.permute(0, 2, 3, 1) * frame.new_tensor((2 / width, 2 / height))
        return F.grid_sample(history, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def detector(device, aligned):
    model = CenterDetector(input_channels=8)
    if aligned:
        model.history_alignment = HistoryAlignment()
    return model.to(device)


def initialize(path, device, aligned):
    model = detector(device, aligned)
    state = torch.load(path, map_location=device, weights_only=True)
    first = state[FIRST_WEIGHT]
    if first.shape[1] != 8:
        raise ValueError("The reference checkpoint must contain seven radar channels and the local contrast channel")
    state[FIRST_WEIGHT] = torch.cat((first[:, :7], torch.zeros_like(first[:, :1]), first[:, 7:]), dim=1)
    mismatch = model.load_state_dict(state, strict=not aligned)
    if mismatch.unexpected_keys or any(not key.startswith("history_alignment.") for key in mismatch.missing_keys):
        raise ValueError("The common initialization does not match the detector backbone")
    return model


def clips(records, features, labels, rng, device):
    batch = np.empty((STEPS, len(records), features.shape[1], TILE, TILE), dtype=np.uint8)
    truth = [[] for _ in range(STEPS)]
    flips = rng.random(len(records)) < 0.5
    for b, (index, left, top) in enumerate(records):
        for step, i in enumerate(range(index - STEPS + 1, index + 1)):
            frame = features[i - 2, :, top:top + TILE, left:left + TILE]
            boxes = target_in_tile(labels[i], left, top)
            if flips[b]:
                frame = frame[..., ::-1]
                boxes[:, [0, 2]] = TILE - boxes[:, [2, 0]]
            batch[step, b] = frame
            truth[step].append({"boxes": torch.from_numpy(boxes)})
    images = torch.from_numpy(batch).to(device).float().div_(255)
    levels = torch.from_numpy(rng.uniform(0.85, 1.15, len(records)).astype(np.float32)).to(device)
    keep_history = torch.from_numpy(rng.random((STEPS, len(records))) >= HISTORY_DROP).to(device)
    return (images * levels[None, :, None, None, None]).clamp_(0, 1), truth, keep_history


def network_input(model, frame, history):
    if history is None:
        history = frame.new_zeros((len(frame), 1, *frame.shape[-2:]))
    else:
        if hasattr(model, "history_alignment"):
            history = model.history_alignment(frame, history)
        history = F.interpolate(history, size=frame.shape[-2:], mode="bilinear", align_corners=False)
    return torch.cat((frame, history), dim=1)


def fit(folder, path, features, labels, *, feedback, aligned=False):
    device = backend()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = initialize(path, device, aligned)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CHECKPOINT_EPOCHS[-1], eta_min=3e-5)
    ids = tuple(i for i in labels if all(t in labels for t in range(i - STEPS + 1, i + 1)))
    _, _, height, width = features.shape
    checkpoints = []
    for epoch in range(1, CHECKPOINT_EPOCHS[-1] + 1):
        records = training_tiles(ids, labels, width, height, rng, epoch, ())
        records.extend((i, *draw_windows(labels[i], width, height, rng)) for i in ids)
        rng.shuffle(records)
        model.train()
        losses = []
        for start in range(0, len(records), BATCH_SIZE):
            images, targets, gate = clips(records[start:start + BATCH_SIZE], features, labels, rng, device)
            history = None
            optimizer.zero_grad(set_to_none=True)
            step_losses = []
            for frame, truth, keep in zip(images, targets, gate, strict=True):
                context = history * keep[:, None, None, None] if history is not None else None
                output = model(network_input(model, frame, context))
                loss = center_loss(model, output, truth)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite detection loss")
                (loss / STEPS).backward()
                if feedback:
                    history = output[:, :1].detach().sigmoid()
                step_losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            losses.append(np.mean(step_losses))
        schedule.step()
        print(f"{folder.name} epoch {epoch}/{CHECKPOINT_EPOCHS[-1]}: {np.mean(losses):.4f}", flush=True)
        if epoch in CHECKPOINT_EPOCHS:
            checkpoint = folder / f"epoch{epoch}.pt"
            torch.save(model.state_dict(), checkpoint)
            checkpoints.append(dict(epoch=epoch, path=checkpoint.name, sha256=sha_file(checkpoint), feedback=feedback,
                                    aligned=aligned, device=device.type, clip_training_frames=len(ids)))
    return checkpoints


@torch.inference_mode()
def predict(folder, checkpoint, features, ids):
    device = backend()
    path = folder / checkpoint["path"]
    if sha_file(path) != checkpoint["sha256"]:
        raise ValueError("Feedback checkpoint checksum differs")
    model = detector(device, checkpoint.get("aligned", False)).eval()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    history = None
    result = {}
    last = None
    for i in ids:
        if last is not None and i != last + 1:
            history = None
        frame = torch.from_numpy(features[i - 2].copy()).to(device).float()[None] / 255
        output = model(network_input(model, frame, history))
        result[i] = decode_center_map(model, output[0], *frame.shape[-2:])
        if checkpoint["feedback"]:
            history = output[:, :1].sigmoid()
        last = i
    return result


def compare_at_fixed_count(predictions, report, development, folder, features):
    before = report["arms"]["control"]["selected"]["operation"]
    after = report["arms"]["feedback"]["selected"]
    control = filter_by_config(with_nms(predictions["control"], before["nms"]), before)
    baseline, base_rows = summarize(control, development)
    outcomes = {}
    disabled = predict(folder, after["checkpoint"] | {"feedback": False}, features, warm_ids(development))
    for name, raw in (("on", predictions["feedback"]), ("disabled", disabled)):
        candidates = with_nms(raw, after["operation"]["nms"])
        limited = {}
        for i in development:
            boxes, scores = candidates[i]
            size = len(control[i][0])
            if len(boxes) < size:
                raise ValueError("Feedback candidates could not reproduce the control output count")
            order = np.argsort(-scores, kind="stable")[:size]
            limited[i] = boxes[order], scores[order], None
        score, rows = summarize(limited, development)
        outcomes[name] = dict(development=score, paired_vs_control=paired_block_f1(rows, base_rows))
    return dict(control=baseline, feedback=outcomes)


def evaluate(output, ffmpeg, *, aligned=False, audit_only=False):
    fit_ids, cal_ids, dev_ids = PROFILES["prospective"]
    gray = load_gray(output, ffmpeg)
    training = load_labels(output, fit_ids, gray)[0]
    calibration = load_labels(output, cal_ids, gray)[0]
    features = feature_cache(output, gray, max(dev_ids), CausalFeatures("stacked"))
    reference = read_json(output / REFERENCE)
    begin = output / reference["config"]["path"]
    if sha_file(begin) != reference["config"]["sha256"] or reference["config"]["training_label_frames"] != len(training):
        raise ValueError("The common initialization is not the recorded early SNL checkpoint")
    standard = output / "feedback-center"
    study = output / ("feedback-center-aligned" if aligned else "feedback-center")
    study.mkdir(exist_ok=True)
    settings = dict(code_sha256=sha_file(Path(__file__)), center_sha256=sha_file(Path(__file__).with_name("snl_center.py")),
                    initialization_sha256=sha_file(begin), seed=SEED, steps=STEPS, history_dropout=HISTORY_DROP,
                    epochs=list(CHECKPOINT_EPOCHS), train_ids=list(fit_ids), train_runtime=str(torch.__version__))
    report = dict(settings=settings, matching_iou=0.50, arms={})
    locations = {"control": standard / "control", "feedback": study / "feedback"}
    if audit_only:
        saved = read_json(study / "comparison.json")
        raw = {name: predict(locations[name], arm["selected"]["checkpoint"], features, warm_ids(dev_ids))
               for name, arm in saved["arms"].items()}
        truth = load_labels(output, dev_ids, gray)[0]
        result = compare_at_fixed_count(raw, saved, truth, locations["feedback"], features)
        write_json(study / "fixed-count.json", result)
        print(result, flush=True)
        return
    cal_steps = warm_ids(cal_ids)
    for name, feedback in (("control", False), ("feedback", True)):
        folder = locations[name]
        use_alignment = aligned and feedback
        folder.mkdir(exist_ok=True)
        registry = folder / "checkpoints.json"
        if registry.is_file():
            saved = read_json(registry)
            fields = ("initialization_sha256", "seed", "steps", "history_dropout", "epochs", "train_ids", "train_runtime")
            if (any(saved["settings"][field] != settings[field] for field in fields)
                    or any(c["feedback"] != feedback or c.get("aligned", False) != use_alignment for c in saved["checkpoints"])):
                raise ValueError("The cached model belongs to another training recipe")
            checkpoints = saved["checkpoints"]
        else:
            checkpoints = fit(folder, begin, features, training, feedback=feedback, aligned=use_alignment)
            saved = dict(settings=settings, checkpoints=checkpoints)
            write_json(registry, saved)
        trials = []
        outputs = {}
        for checkpoint in checkpoints:
            raw = predict(folder, checkpoint, features, cal_steps)
            operation, score, _ = select_operation(raw, calibration)
            outputs[checkpoint["epoch"]] = raw
            trials.append(dict(checkpoint=checkpoint, operation=operation, calibration=score))
        choice = max(trials, key=lambda result: quality(result["calibration"]))
        short = [i for i in cal_steps if i <= cal_ids[1]]
        prefix = predict(folder, choice["checkpoint"], features[:cal_ids[1] - 1], short)
        for i in short:
            for before, after in zip(prefix[i], outputs[choice["checkpoint"]["epoch"]][i], strict=True):
                np.testing.assert_array_equal(before, after)
        report["arms"][name] = dict(selected=choice, trials=trials, prefix_checked_frames=short,
                                    training_code_sha256=saved["settings"]["code_sha256"])
        print(name, "calibration", choice["calibration"], flush=True)
    write_json(study / "calibration.json", report)

    predictions = {name: predict(locations[name], arm["selected"]["checkpoint"], features, warm_ids(dev_ids))
                   for name, arm in report["arms"].items()}
    development = load_labels(output, dev_ids, gray)[0]
    released = reference["config"]
    raw = load_candidates(output, released, "development")
    reference_metrics, previous = summarize(filter_by_config(with_nms(raw, released["nms"]), released), development)
    if reference_metrics != reference["development"]:
        raise ValueError("The released comparison did not replay")
    frame_rows = {"released": previous}
    for name, arm in report["arms"].items():
        point = arm["selected"]
        ops = point["operation"]
        raw = predictions[name]
        result, rows = summarize(filter_by_config(with_nms(raw, ops["nms"]), ops), development)
        arm.update(development=result, development_gate_passed=passed(point["calibration"]) and passed(result),
                   all_candidate_matches=sum(detection_counts(raw[i][0], truth)[0] for i, truth in development.items()))
        frame_rows[name] = rows
        print(name, "development", result, flush=True)
    report.update(released_development=reference_metrics,
                  paired_f1_feedback_vs_control=paired_block_f1(frame_rows["feedback"], frame_rows["control"]))
    write_json(study / "comparison.json", report)
    write_csv(study / "development.csv", [dict(method=name, **row) for name, rows in frame_rows.items() for row in rows])
    audit = compare_at_fixed_count(predictions, report, development, locations["feedback"], features)
    write_json(study / "fixed-count.json", audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--align-history", action="store_true")
    parser.add_argument("--audit-fixed-count", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    evaluate(args.output.resolve(), args.ffmpeg, aligned=args.align_history, audit_only=args.audit_fixed_count)
