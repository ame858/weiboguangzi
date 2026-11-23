"""Test an early-frame route prior after registering each current SNL image."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from experiment import filter_by_config, load_candidates, quality, select_operation, summarize, with_nms
from route_prior import centers
from snl_bytetrack import REFERENCE
from snl_detector_agreement import combine
from snl_inputs import DEFAULT_OUTPUT, PROFILES, load_gray, load_labels, read_json, write_json
from videosar.snl_causal import matched_boxes


RADII = (10, 20, 30, 40, 55, 75)


class ReferenceRegistration:
    def __init__(self, image):
        self.extractor = cv2.SIFT_create(nfeatures=1200, contrastThreshold=0.012)
        self.equalizer = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.points, self.descriptors = self.describe(image)
        if self.descriptors is None or len(self.descriptors) < 12:
            raise ValueError("The radar reference frame has insufficient spatial features")

    def describe(self, image):
        return self.extractor.detectAndCompute(self.equalizer.apply(image), None)

    def estimate(self, image):
        return self.estimate_features(self.describe(image), (self.points, self.descriptors))

    def estimate_features(self, source, reference):
        points, descriptors = source
        reference_points, reference_descriptors = reference
        if reference_descriptors is None:
            return None, 0
        if descriptors is None:
            return None, 0
        pairs = [pair[0] for pair in self.matcher.knnMatch(descriptors, reference_descriptors, k=2)
                 if len(pair) == 2 and pair[0].distance < 0.77 * pair[1].distance]
        if len(pairs) < 12:
            return None, 0
        origin = np.float32([points[pair.queryIdx].pt for pair in pairs])
        destination = np.float32([reference_points[pair.trainIdx].pt for pair in pairs])
        matrix, inliers = cv2.estimateAffinePartial2D(origin, destination, method=cv2.RANSAC,
                                                     ransacReprojThreshold=2.5, maxIters=3000, confidence=0.995)
        count = int(inliers.sum()) if inliers is not None else 0
        return (matrix if count >= 12 else None), count


def project(points, matrix):
    return points @ matrix[:, :2].T + matrix[:, 2]


def registration(gray, anchor, ids):
    reference = ReferenceRegistration(gray[anchor - 2])
    inliers, transforms = {}, {anchor: np.eye(2, 3)}
    for index in sorted(set(ids) - {anchor}):
        matrix, count = reference.estimate(gray[index - 2])
        inliers[index] = count
        if matrix is not None:
            transforms[index] = matrix
    return transforms, inliers


def chained_registration(gray, anchor, ids):
    """Bridge failed direct matches through nearby frames already registered to the anchor."""
    reference = ReferenceRegistration(gray[anchor - 2])
    transforms = {anchor: np.eye(2, 3)}
    keypoints = {anchor: (reference.points, reference.descriptors)}
    inliers = {}
    earlier = sorted((i for i in ids if i < anchor), reverse=True)
    later = sorted(i for i in ids if i > anchor)
    for block, direction in ((earlier, 1), (later, -1)):
        for index in block:
            source = reference.describe(gray[index - 2])
            keypoints[index] = source
            for neighbor in dict.fromkeys((anchor, *(index + direction * hop for hop in (12, 6, 3, 1)))):
                if neighbor not in transforms or (direction < 0 and neighbor >= index):
                    continue
                local, count = reference.estimate_features(source, keypoints[neighbor])
                if local is not None:
                    transforms[index] = transforms[neighbor] @ np.vstack((local, (0, 0, 1)))
                    inliers[index] = count
                    break
    return transforms, inliers


def route_distances(boxes, transforms, prior):
    result = {}
    for index, values in boxes.items():
        points = centers(values)
        # Missing registration leaves the detector's output scores unchanged for the frame.
        result[index] = (prior.query(project(points, transforms[index]))[0] if index in transforms
                         else np.zeros(len(points)))
    return result


def change_scores(raw, distances, radius, shape):
    if radius is None:
        return raw
    output = {}
    for index, (boxes, scores) in raw.items():
        distance = distances[index]
        if shape == "hard":
            visible = distance <= radius
            output[index] = boxes[visible], scores[visible]
        else:
            output[index] = boxes, (scores * np.exp(-0.5 * (distance / radius)**2)).astype(np.float32)
    return output


def compare_routes(output, ffmpeg):
    train_ids, cal_ids, dev_ids = PROFILES["prospective"]
    anchor = train_ids[-1]
    source_ids = tuple(sorted(set(train_ids[::5]) | {anchor}))
    gray = load_gray(output, ffmpeg)
    source, calibration, development = [load_labels(output, ids, gray)[0] for ids in (source_ids, cal_ids, dev_ids)]
    release = read_json(output / REFERENCE)
    agreement = read_json(output / "center-retinanet-agreement.json")
    same_scope = (calibration, development)
    raw = {}
    for split in ("validation", "development"):
        center, retina = [load_candidates(output, config, split) for config in (release["config"], agreement["second"])]
        fused = combine(center, retina, agreement["first"]["threshold"], agreement["second"]["threshold"],
                        agreement["selected"]["agreement_boost"], agreement["selected"]["second_weight"])
        raw[split] = {"released": center, "retina_agreement": fused}
    alignment_ids = set(source_ids) | set(raw["validation"]["released"]) | set(raw["development"]["released"])
    matrices, inliers = registration(gray, anchor, alignment_ids)
    prior = cKDTree(np.concatenate([project(centers(boxes), matrices[index])
                                    for index, boxes in source.items() if index in matrices]))
    unaligned = cKDTree(np.concatenate([centers(boxes) for index, boxes in source.items() if index in matrices]))
    coverage = {}
    for name, truth in zip(("validation", "development"), same_scope, strict=True):
        valid = {i: boxes for i, boxes in truth.items() if i in matrices}
        aligned_distances = np.concatenate(list(route_distances(valid, matrices, prior).values()))
        nominal_distances = unaligned.query(centers(np.concatenate(list(valid.values()))))[0]
        coverage[name] = {"registered_frames": len(valid), "frames": len(truth), "boxes_in_registered_frames": len(aligned_distances),
                          "unaligned": {r: float(np.mean(nominal_distances <= r)) for r in RADII},
                          "aligned": {r: float(np.mean(aligned_distances <= r)) for r in RADII}}

    outcomes = {}
    for method in raw["validation"]:
        val, dev = (raw[split][method] for split in ("validation", "development"))
        val_distances = route_distances({i: boxes for i, (boxes, _) in val.items()}, matrices, prior)
        dev_distances = route_distances({i: boxes for i, (boxes, _) in dev.items()}, matrices, prior)
        trials = []
        choices = [(None, None), *((r, shape) for r in RADII for shape in ("hard", "soft"))]
        for radius, shape in choices:
            operation, summary, _ = select_operation(change_scores(val, val_distances, radius, shape), calibration)
            trials.append({"radius_px": radius, "shape": shape, "operation": operation, "validation": summary})
        selected = max(trials, key=lambda trial: quality(trial["validation"]))
        selected = selected.copy()
        raw_predictions = change_scores(dev, dev_distances, selected["radius_px"], selected["shape"])
        point = selected["operation"]
        selected["development"], _ = summarize(filter_by_config(with_nms(raw_predictions, point["nms"]), point), development)
        selected["max_matches_from_remaining_candidates"] = sum(
            len(matched_boxes(raw_predictions[index][0], truth)) for index, truth in development.items())
        outcomes[method] = {"selected": selected, "trials": trials}
        print(method, selected, flush=True)
    report = {"anchor": anchor, "fit_ids": list(source_ids), "registration_inliers": inliers,
              "fit_frames_registered": sum(i in matrices for i in source_ids), "coverage": coverage,
              "reference_development": release["development"], "comparisons": outcomes}
    write_json(output / "registered-route-comparison.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prospective")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    cv2.setNumThreads(4)
    cv2.setRNGSeed(1729)
    compare_routes(args.output.resolve(), args.ffmpeg)
