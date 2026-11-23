"""Compare the pinned upstream ByteTrack with the same SNL detector and scores."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audit_detection import paired_block_f1
from experiment import filtered, filter_by_config, load_candidates, nonmax, quality, summarize, warm_ids, with_nms
from snl_inputs import DEFAULT_OUTPUT, HEIGHT, PROFILES, WIDTH, load_gray, load_labels, read_json, write_csv, write_json
from snl_yolo import YOLO_VERSION, yolo_runtime

REFERENCE = "decision-radar-center-stacked-epoch36-contrast1.35-support3.json"


def track(candidates, settings):
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker

    settings = SimpleNamespace(**settings)
    tracker = BYTETracker(settings, frame_rate=30)
    predictions = {}
    for index, (boxes, scores) in candidates.items():
        values = np.column_stack((boxes, scores, np.zeros_like(scores)))
        rows = tracker.update(Boxes(values, (HEIGHT, WIDTH))).reshape(-1, 8)
        predictions[index] = rows[:, :4], rows[:, 5], rows[:, 4].astype(int)
    return predictions


def retain_strong(candidates, tracks, high, nms):
    result = {}
    for index, (boxes, scores) in candidates.items():
        strong = scores >= high
        supported_boxes, supported_scores, _ = tracks[index]
        weak = supported_scores < high
        selected_boxes = np.concatenate((boxes[strong], supported_boxes[weak]))
        selected_scores = np.concatenate((scores[strong], supported_scores[weak]))
        boxes_after_nms, scores_after_nms = nonmax(selected_boxes, selected_scores, nms)
        result[index] = boxes_after_nms, scores_after_nms, np.full(len(scores_after_nms), -1)
    return result


def compare(output, decision, ffmpeg):
    yolo_runtime(output)
    config = decision["config"]
    gray = load_gray(output, ffmpeg)
    _, cal_ids, dev_ids = PROFILES["prospective"]
    cal = load_labels(output, cal_ids, gray)[0]
    dev = load_labels(output, dev_ids, gray)[0]
    calibration = with_nms(load_candidates(output, config, "validation"), config["nms"])
    development = with_nms(load_candidates(output, config, "development"), config["nms"])
    for split, predictions in ((cal, calibration), (dev, development)):
        if tuple(predictions) != tuple(warm_ids(split)) or max(predictions) - min(predictions) + 1 != len(predictions):
            raise ValueError("The tracker needs consecutive candidates including the full warmup")

    control, control_rows = summarize(filter_by_config(development, config), dev)
    if control != decision["development"]:
        raise ValueError("The cached scores do not reproduce the detector reference")
    upstream = {"track_high_thresh": 0.25, "track_low_thresh": 0.10, "new_track_thresh": 0.25,
                "track_buffer": 30, "match_thresh": 0.8, "fuse_score": True}
    grid = [upstream]
    thresholds = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
    for value in thresholds:
        for fuse in (True, False):
            grid.append(upstream | {"track_high_thresh": value, "new_track_thresh": value,
                                    "track_low_thresh": 0.5 * value, "fuse_score": fuse})
    calibration_tracks = [track(calibration, params) for params in grid]
    sweep = [params | summarize(predicted, cal)[0] for params, predicted in zip(grid, calibration_tracks, strict=True)]
    support_sweep = [params | summarize(retain_strong(calibration, predicted, params["track_high_thresh"], config["nms"]), cal)[0]
                     for params, predicted in zip(grid, calibration_tracks, strict=True)]
    best = max(range(len(sweep)), key=lambda i: quality(sweep[i]))
    params = grid[best]
    full = track(development, params)
    prefix = track({i: development[i] for i in tuple(development)[:len(development) // 2]}, params)
    for i, arrays in prefix.items():
        for replayed, actual in zip(arrays, full[i], strict=True):
            np.testing.assert_array_equal(replayed, actual)
    result, rows = summarize(full, dev)
    selected_threshold = max(thresholds, key=lambda value: quality(summarize(filtered(calibration, value, None), cal)[0]))
    support_index = max(range(len(support_sweep)), key=lambda i: quality(support_sweep[i]))
    support = grid[support_index]
    supported = retain_strong(development, track(development, support), support["track_high_thresh"], config["nms"])
    support_result, support_rows = summarize(supported, dev)
    report = {
        "implementation": f"Ultralytics {YOLO_VERSION} BYTETracker", "source_detector": config,
        "settings": params, "calibration": {key: value for key, value in sweep[best].items() if key not in params},
        "development": result, "reference_detector": control, "frame_prefix_identical": True,
        "default_bytetrack_development": summarize(track(development, upstream), dev)[0],
        "calibration_only_detection_threshold": selected_threshold,
        "threshold_only_development": summarize(filtered(development, selected_threshold, None), dev)[0],
        "delta_from_reference_f1": result["f1"] - control["f1"],
        "paired_uncertainty": paired_block_f1(rows, control_rows),
        "retain_strong": {"settings": support, "calibration": support_sweep[support_index], "development": support_result,
                          "delta_from_reference_f1": support_result["f1"] - control["f1"],
                          "paired_uncertainty": paired_block_f1(support_rows, control_rows)},
    }
    write_json(output / "bytetrack-development.json", report)
    write_csv(output / "bytetrack-calibration.csv", sweep)
    write_csv(output / "bytetrack-development-frames.csv", rows)
    write_csv(output / "bytetrack-retain-strong-calibration.csv", support_sweep)
    write_csv(output / "bytetrack-retain-strong-development-frames.csv", support_rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    output = args.output.resolve()
    decision = read_json(output / REFERENCE)
    print(compare(output, decision, args.ffmpeg), flush=True)


if __name__ == "__main__":
    main()
