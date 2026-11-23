from collections import Counter

import numpy as np

from .analysis import score_masks, target_summary
from .center_box import BOX_WEIGHT, CENTER_BIAS, CENTER_OUTPUTS, center_masks
from .contrast_study import (AREAS, BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED as SOURCE_SEED, THRESHOLDS,
                             TRAIN_SEED, locked_specs, paired_and_aperture, save_contrast_scene, score_validation,
                             with_contrast_stem)
from .data import read_json, write_json
from .echo_attribution import file_hash, frozen_settings
from .experiment import read_split
from .open_compare import HISTORICAL, SOURCES, comparison_gate, evaluate_models
from .small_shadow import CONDITIONS, OPERATING_FIELDS, operating_rank, recall_for, tune
from .stress import MATCH
from .study import read_registered
from .temporal_study import SIZE_BANDS, write_size_breakdown
from .unet import create_model, fit, load_weights, predict_centers, runtime
from .visuals import write_csv


SEED = 101651
CENTER_THRESHOLDS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                     0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def init_center_model(torch, smp, parent):
    expanded = with_contrast_stem(torch, smp, parent)
    model = create_model(torch, smp, True, local_contrast=True, center_head=True)
    model.to(memory_format=torch.channels_last)
    weights = expanded.state_dict()
    key = "segmentation_head.0"
    kernel = weights[f"{key}.weight"]
    new_kernel = kernel.new_zeros((CENTER_OUTPUTS - 1, *kernel.shape[1:]))
    weights[f"{key}.weight"] = torch.cat((kernel, new_kernel))
    prior = weights[f"{key}.bias"]
    bias = prior.new_tensor((CENTER_BIAS, *([np.log1p(2)] * 4)))
    weights[f"{key}.bias"] = torch.cat((prior, bias))
    model.load_state_dict(weights)
    return model


def tune_centers(torch, model, settings, validation, normal_reference):
    records = []
    for sample in validation:
        heatmap, distances = predict_centers(torch, model, sample["aligned"], settings, "cpu", BATCH_SIZE)
        boxes = center_masks(heatmap, distances, 0, 1)
        scores = score_masks(boxes, sample["truth"], MATCH)["frames"]
        for peak, record in zip(heatmap.max(axis=(1, 2)), scores):
            records.append((float(peak), sample["entry"]["group"], record["target_visible"],
                            record["best_component_iou"] >= MATCH))
    rows = []
    for threshold in CENTER_THRESHOLDS:
        hits = Counter(group for peak, group, _, match in records if peak >= threshold and match)
        support = Counter(group for _, group, visible, _ in records if visible)
        predicted = sum(peak >= threshold for peak, _, _, _ in records)
        tp = sum(hits.values())
        fp = predicted - tp
        rows.append(dict(threshold=threshold, min_area=1, closing=False, false_positive=fp,
                         normal_recall=hits["normal"] / support["normal"], small_recall=hits["small"] / support["small"],
                         **target_summary(tp, fp, sum(support.values()) - tp)))
    return max(rows, key=lambda row: operating_rank(row, normal_reference)), rows


def rectangle_oracle(mask):
    """Best IoU over all axis-aligned rectangles for a nonempty fixed target."""
    ys, xs = np.nonzero(mask)
    cropped = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    height, width = cropped.shape
    integral = np.pad(cropped, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    y0, y1 = np.triu_indices(height, k=0)
    x0, x1 = np.triu_indices(width, k=0)
    y1, x1 = y1 + 1, x1 + 1
    intersection = (integral[y1[:, None], x1] - integral[y0[:, None], x1]
                    - integral[y1[:, None], x0] + integral[y0[:, None], x0])
    area = (y1 - y0)[:, None] * (x1 - x0)
    return float(len(ys) / cropped.size), float(np.max(intersection / (area + len(ys) - intersection)))


def write_box_diagnostics(root, test, torch, correlate, model, settings):
    records = []
    for sample in read_registered(test, correlate):
        group = sample["entry"]["group"]
        if group == "empty":
            continue
        heatmap, distances = predict_centers(torch, model, sample["aligned"], settings, "cpu", BATCH_SIZE)
        boxes = center_masks(heatmap, distances, 0, 1)
        frame_scores = score_masks(boxes, sample["truth"], MATCH)["frames"]
        for i, (truth, frame, peak) in enumerate(zip(sample["truth"], frame_scores, heatmap.max(axis=(1, 2)))):
            if not frame["target_visible"]:
                continue
            area = int(truth.sum())
            band = SIZE_BANDS[np.digitize(area, (12, 24, 48))]
            bounding_iou, oracle_iou = rectangle_oracle(truth)
            records.append(dict(scene_id=sample["entry"]["scene_id"], group=group, frame=i,
                                label_area=area, band=band, bounding_box_iou=bounding_iou, best_rectangle_iou=oracle_iou,
                                predicted_box_iou=frame["best_component_iou"], peak_probability=float(peak)))
    summary = []
    for key in ("all", "small", "normal", *SIZE_BANDS):
        subset = [r for r in records if key in ("all", r["group"], r["band"])]
        summary.append(dict(group=key, visible_frames=len(subset),
                            **{f"{metric}_matches": sum(row[metric] >= MATCH for row in subset)
                               for metric in ("bounding_box_iou", "best_rectangle_iou", "predicted_box_iou")},
                            accepted_matches=sum(row["predicted_box_iou"] >= MATCH
                                                 and row["peak_probability"] >= settings["threshold"] for row in subset)))
    write_csv(root / "box_diagnostics.csv", records)
    write_csv(root / "box_diagnostics_summary.csv", summary)
    return summary


def run_center_study(root, training_run):
    torch, smp, correlate = runtime()
    torch.set_num_threads(min(6, torch.get_num_threads()))
    source = training_run / "dataset"
    previous = read_json(training_run / "protocol.json")
    frozen = read_json(training_run / "frozen_validation.json")
    if (previous["seed"] != SOURCE_SEED or previous["train_seed"] != TRAIN_SEED or previous["epochs"] != EPOCHS
            or previous["batch_size"] != BATCH_SIZE or previous["learning_rate"] != LEARNING_RATE
            or frozen["dataset_sha256"] != file_hash(source / "dataset.json")):
        raise ValueError("center study requires the locked local-contrast data and matching training budget")
    control_path = training_run / "models/local_contrast.json"
    control = read_json(control_path)
    if control["weight_sha256"] != frozen["candidates"]["local_contrast"]["weight_sha256"]:
        raise ValueError("local-contrast checkpoint disagrees with the validation freeze")
    partitions = read_split(source)
    settings, checkpoints, operating, hashes = frozen_settings()
    if any(previous[key] != value for key, value in hashes.items()):
        raise ValueError("frozen baselines differ from those used in the contrast study")
    native = settings["unet"]
    test_specs = []
    for spec in locked_specs(SEED):
        if spec["split"] == "test":
            name = spec["scene_id"].replace("contrast-", "center-", 1)
            test_specs.append(spec | {"scene_id": name, "scene_family": name})
    write_json(root / "protocol.json", dict(seed=SEED, train_seed=TRAIN_SEED, test_scenes=test_specs,
        source_dataset_sha256=frozen["dataset_sha256"], control_weight_sha256=control["weight_sha256"],
        epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, **hashes,
        target="unit Gaussian at integer midpoint of label box, sigma=1 pixel; four log1p distances at center",
        loss="mask BCE+Dice plus center focal loss and Smooth L1 box loss", box_loss_weight=BOX_WEIGHT,
        center_thresholds=CENTER_THRESHOLDS, mask_thresholds=THRESHOLDS, mask_areas=AREAS, match_iou=MATCH,
        versions={"torch": torch.__version__, "smp": smp.__version__}, source_projects=SOURCES,
        gate={"precision": 0.8, "recall": 0.8, "small_recall_gain": 0.10, "normal_recall_max_drop": 0.05}))
    test = []
    for spec in test_specs:
        test.append(save_contrast_scene(root / "dataset", spec))
        if len(test) % 12 == 0:
            print(f"Generated and checked {len(test)}/36 locked SAR sequences", flush=True)
    if (Counter(row["condition"] for _, row in test) != dict.fromkeys(CONDITIONS, 9)
            or Counter(row["pulses"] for _, row in test) != {61: 18, 91: 18}):
        raise ValueError("locked test condition counts differ from the protocol")
    manifest = root / "dataset/dataset.json"
    write_json(manifest, {"profile": "center-study", "seed": SEED, "entries": [row for _, row in test]})
    write_csv(root / "dataset/manifest.csv", [row for _, row in test])
    train = read_registered(partitions["train"], correlate)
    validation = read_registered(partitions["validation"], correlate)
    native_model = load_weights(torch, smp, checkpoints["unet"], native, "cpu")
    fpn = load_weights(torch, smp, checkpoints["fpn"], settings["fpn"], "cpu")
    control_model = load_weights(torch, smp, control_path.with_name(control["checkpoint"]), control, "cpu")
    baseline_scores = score_validation(torch, native_model, native, validation)
    normal_reference = recall_for(baseline_scores, validation, operating["unet"], ("normal",))
    print("Training joint center, box and mask outputs on the locked training split", flush=True)
    model, trained = fit(torch, smp, [(s["aligned"], s["truth"]) for s in train],
                         [(s["aligned"], s["truth"]) for s in validation], temporal=True, augment_noise=False,
                         epochs=EPOCHS, seed=TRAIN_SEED, device="cpu", batch_size=BATCH_SIZE,
                         initialization=(init_center_model(torch, smp, native_model), native),
                         learning_rate=LEARNING_RATE, local_contrast=True, center_head=True)
    mask_scores = score_validation(torch, model, trained, validation)
    mask_point, mask_grid = tune(mask_scores, validation, normal_reference, THRESHOLDS, AREAS)
    box_point, box_grid = tune_centers(torch, model, trained, validation, normal_reference)
    write_csv(root / "validation_joint_mask.csv", mask_grid)
    write_csv(root / "validation_center_box.csv", box_grid)
    model_dir = root / "models"
    model_dir.mkdir()
    checkpoint = model_dir / "center_box.pt"
    torch.save(model.state_dict(), checkpoint)
    trained.update(checkpoint=checkpoint.name, weight_sha256=file_hash(checkpoint), registered=True,
                   parent_weight_sha256=native["weight_sha256"])
    mask_settings = trained | {key: mask_point[key] for key in OPERATING_FIELDS}
    box_settings = trained | {key: box_point[key] for key in OPERATING_FIELDS} | {"detection_mode": "center_box"}
    write_json(model_dir / "joint_mask.json", mask_settings)
    write_json(model_dir / "center_box.json", box_settings)
    candidates = {"joint_mask": mask_point, "center_box": box_point}
    choice = max(candidates, key=lambda key: operating_rank(candidates[key], normal_reference))
    write_json(root / "frozen_validation.json", dict(dataset_sha256=file_hash(manifest), selected_candidate=choice,
               weight_sha256=trained["weight_sha256"], candidates=candidates, normal_reference=normal_reference))
    print("Validation operating points frozen. Evaluating six methods on the locked test set.", flush=True)
    deployed = load_weights(torch, smp, checkpoint, box_settings, "cpu")
    registry = {HISTORICAL: (native_model, native), "local_contrast": (control_model, control),
                "fpn": (fpn, settings["fpn"]), "difference": (None, operating["difference"]),
                "joint_mask": (deployed, mask_settings), "center_box": (deployed, box_settings)}
    groups = evaluate_models(root, test, registry, torch, correlate)
    write_size_breakdown(root, test, registry)
    paired = paired_and_aperture(root, registry, (("joint_mask", "local_contrast"), ("center_box", "local_contrast"),
        ("center_box", "joint_mask"), ("joint_mask", HISTORICAL), ("center_box", HISTORICAL)))
    boxes = write_box_diagnostics(root, test, torch, correlate, deployed, box_settings)
    table = {(row["method"], row["group"]): row for row in groups}
    write_json(root / "results.json", dict(groups=groups, paired=paired, box_diagnostics=boxes,
               selected_candidate=choice, gates={name: comparison_gate(table, name) for name in candidates}))
