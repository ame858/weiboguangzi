"""Calibrate agreement between the center detector and an open-source detector."""
import argparse
from itertools import product
from pathlib import Path

import numpy as np

from experiment import diagnostic_coverage, filtered, load_candidates, quality, select_operation, summarize, with_nms
from snl_inputs import DEFAULT_OUTPUT, PROFILES, load_gray, load_labels, read_json, write_json
from videosar.snl_causal import box_iou


def combine(left, right, first_threshold, second_threshold, boost, second_weight):
    result = {}
    for index in left:
        a, a_scores = left[index]
        b, b_scores = right[index]
        first = a_scores / (a_scores + first_threshold)
        second = b_scores / (b_scores + second_threshold / second_weight)
        shared = box_iou(a, b) >= 0.30
        b_support = (shared * second[None]).max(axis=1, initial=0)
        a_support = (shared * first[:, None]).max(axis=0, initial=0)
        a_confidence = first + boost * b_support * (1 - first)
        b_confidence = second + boost * a_support * (1 - second)
        result[index] = np.concatenate((a, b)), np.concatenate((a_confidence, b_confidence))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--report", type=Path, help="Write this comparison separately from the default report")
    args = parser.parse_args()
    output = args.output.resolve()
    first, second = (read_json(path)["config"] for path in (args.first, args.second))
    gray = load_gray(output, args.ffmpeg)
    profile = read_json(output / "split-protocol.json")["profile"]
    _, va, dv = PROFILES[profile]
    validation, _, _ = load_labels(output, va, gray)
    development, _, _ = load_labels(output, dv, gray)
    cal = tuple(load_candidates(output, cfg, "validation") for cfg in (first, second))
    dev = tuple(load_candidates(output, cfg, "development") for cfg in (first, second))
    strengths = (first["threshold"], second["threshold"])
    trials = []
    for boost, weight in product((0.0, 0.25, 0.50), (0.80, 1.0, 1.25)):
        output_cal = combine(*cal, *strengths, boost, weight)
        operation, metrics, _ = select_operation(output_cal, validation)
        trials.append({"agreement_boost": boost, "second_weight": weight, "operation": operation, "validation": metrics})
    chosen = max(trials, key=lambda v: quality(v["validation"]))
    raw = combine(*dev, *strengths, chosen["agreement_boost"], chosen["second_weight"])
    op = chosen["operation"]
    pooled = with_nms(raw, op["nms"])
    predicted = filtered(pooled, op["threshold"], op["low"])
    chosen["development"], _ = summarize(predicted, development)
    chosen["development_recall_at_60"], chosen["development_center_coverage_at_60"] = diagnostic_coverage(pooled, development)
    report = args.report or output / "center-yolo-agreement.json"
    write_json(report, {"first": first, "second": second, "selected": chosen, "trials": trials})
    print(chosen, flush=True)


if __name__ == "__main__":
    main()
