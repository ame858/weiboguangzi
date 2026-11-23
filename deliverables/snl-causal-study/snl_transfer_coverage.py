"""Measure whether the transfer checkpoints add boxes to the existing dense candidate pool."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from experiment import load_candidates
from snl_inputs import DEFAULT_OUTPUT, PROFILES, load_gray, load_labels, read_json, write_csv, write_json
from snl_local_shadow_proposals import proposals
from videosar.snl_causal import matched_boxes


def audit(output, ffmpeg):
    study = output / "external-background-transfer"
    transfer = read_json(study / "comparison.json")
    archive = read_json(output / "forward-dense-frame-ranking/comparison.json")
    dense_config = archive["evaluation_detector"]
    if dense_config["dense_candidate_limit"] != 4096:
        raise ValueError("Expected the locked 4,096-box center pool")
    features = np.load(output / "motion-integration/motion-integration.npy", mmap_mode="r", allow_pickle=False)
    folders = {"external": study / "external", "early-snl": study / "early-snl", "random": output / "simulation-pretraining"}
    _, cal_ids, dev_ids = PROFILES["prospective"]
    gray = load_gray(output, ffmpeg)
    rows = []
    totals = {}
    for scope, ids in (("validation", cal_ids), ("development", dev_ids)):
        dense = load_candidates(output, dense_config, scope)
        additions = {name: load_candidates(directory, transfer["arms"][name]["selected"]["config"], scope)
                     for name, directory in folders.items()}
        truth = load_labels(output, ids, gray)[0]
        scope_rows = []
        for i, boxes in truth.items():
            initial = np.concatenate((dense[i][0], proposals(features[i - 2])[0]))
            count = len(matched_boxes(initial, boxes))
            expanded = {name: len(matched_boxes(np.concatenate((initial, raw[i][0])), boxes))
                        for name, raw in additions.items()}
            scope_rows.append(dict(split=scope, annotation_id=i, targets=len(boxes), dense_pool=count, **expanded))
        rows.extend(scope_rows)
        totals[scope] = {name: sum(row[name] for row in scope_rows) for name in ("targets", "dense_pool", *folders)}
    if totals["development"]["dense_pool"] != archive["coverage"]["union_before_scoring"]:
        raise ValueError("The archived candidate coverage did not replay")
    report = dict(criterion="one-to-one IoU at least 0.50, ignoring confidence",
                  base="4,096 center proposals plus multiscale local proposals", splits=totals,
                  checkpoint_sha256={name: arm["selected"]["config"]["sha256"] for name, arm in transfer["arms"].items()})
    write_json(study / "candidate-coverage.json", report)
    write_csv(study / "candidate-coverage.csv", rows)
    print(totals, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    cv2.setNumThreads(3)
    audit(args.output.resolve(), args.ffmpeg)
