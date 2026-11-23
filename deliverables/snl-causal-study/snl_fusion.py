"""Cross-fit candidate ranking and box regression on the prospective calibration segment."""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np

from experiment import diagnostic_coverage, filtered, load_candidates, model_key, nonmax, quality, select_operation, summarize
from experiment import with_nms
from snl_detector import FULL_FRAME_ARCHITECTURES
from snl_inputs import DEFAULT_OUTPUT, PROFILES, feature_cache, load_gray, load_labels, read_json, sha_file, write_json
from videosar.snl_causal import CausalFeatures, box_iou

SOURCE_CAP = 350
POOL_CAP = 1250


def family(cfg):
    return tuple(cfg.get(k) for k in ("architecture", "method", "scale", "training_copy_paste",
                                      "training_motion_synthesis", "input_profile", "dense_offsets", "tiled"))


def choose_sources(output, include_full_frame=False):
    cohorts = {}
    for file in sorted(output.glob("decision-*.json")):
        result = read_json(file)
        cfg = result["config"]
        if "path" not in cfg:
            continue
        key = family(cfg)
        current = cohorts.get(key)
        if current is None or quality(result["validation"]) > quality(current["validation"]):
            cohorts[key] = result
    ranked = sorted(cohorts.values(), key=lambda r: quality(r["validation"]), reverse=True)
    selected = [r["config"] for r in ranked[:9]]
    if include_full_frame:
        for architecture in FULL_FRAME_ARCHITECTURES:
            available = [r["config"] for r in ranked if r["config"].get("architecture") == architecture]
            if available and available[0] not in selected:
                selected.append(available[0])
    return selected


def box_means(integral, boxes):
    x0,y0,x1,y1 = np.rint(boxes).astype(int).T
    x0,x1 = np.clip(x0,0,integral.shape[1]-1),np.clip(x1,0,integral.shape[1]-1)
    y0,y1 = np.clip(y0,0,integral.shape[0]-1),np.clip(y1,0,integral.shape[0]-1)
    total = integral[y1,x1] - integral[y1,x0] - integral[y0,x1] + integral[y0,x0]
    return total / np.maximum((x1-x0)*(y1-y0),1)


def measurement(frame, boxes):
    sizes = boxes[:,2:]-boxes[:,:2]
    widths = sizes[:,0]
    heights = sizes[:,1]
    center = (boxes[:,:2]+boxes[:,2:])/2
    inner = np.column_stack((center-0.30*sizes, center+0.30*sizes))
    outer = boxes + np.array([-5,-5,5,5])
    left = np.column_stack((boxes[:,0]-4,boxes[:,1],boxes[:,0],boxes[:,3]))
    right = np.column_stack((boxes[:,2],boxes[:,1],boxes[:,2]+4,boxes[:,3]))
    result = [widths,heights,heights/(1+widths)]
    for channel in frame:
        integral = np.pad(channel,((1,0),(1,0))).cumsum(0,dtype=np.float64).cumsum(1)/255
        core = box_means(integral,inner)
        full = box_means(integral,boxes)
        side_a,side_b = box_means(integral,left),box_means(integral,right)
        result.extend((core,full,box_means(integral,outer)-full,side_a-full,side_b-full))
    return np.column_stack(result).astype(np.float32)


def collect(frame, observations):
    limited = []
    for boxes,scores in observations:
        ids = np.argsort(-scores,kind="stable")[:SOURCE_CAP]
        boxes,scores = boxes[ids],scores[ids]
        limited.append((boxes,scores))
    pooled = np.concatenate([b for b,_ in limited])
    confidence = np.concatenate([s for _,s in limited])
    # Preserve different plausible sizes at the same center for geometric refinement.
    pooled,_ = nonmax(pooled,confidence,0.68)
    pooled = pooled[:POOL_CAP]
    support = []
    for boxes,scores in limited:
        intersect = box_iou(pooled,boxes)
        support.append((intersect >= 0.30, scores))
    votes = np.column_stack([(m*s[None]).max(axis=1,initial=0) for m,s in support])
    cover = np.column_stack([m.any(axis=1) for m,_ in support])
    shape = measurement(frame,pooled)
    data = np.column_stack((shape,votes,cover.sum(axis=1),votes.max(axis=1,initial=0),votes.mean(axis=1)))
    return pooled,data


def make_records(observations, ground_truth, features):
    ids = sorted(set.intersection(*(set(x) for x in observations)))
    measured = {}
    for index in ids:
        measured[index] = collect(features[index-2], [r[index] for r in observations])
    samples = {}
    for index, target in ground_truth.items():
        boxes,data = measured[index]
        overlaps = box_iou(boxes,target)
        high = overlaps.max(axis=1,initial=0)
        if len(target):
            nearest = target[overlaps.argmax(axis=1)]
        else:
            nearest = np.zeros_like(boxes)
        samples[index] = data,boxes,high,nearest
    return measured,samples


def fit(samples, ids):
    x,boxes,iou,near = (np.concatenate([samples[i][j] for i in ids]) for j in range(4))
    target = iou >= 0.5
    options = dict(n_estimators=120,num_leaves=15,max_depth=5,learning_rate=0.05,
                   min_child_samples=22,random_state=331,n_jobs=7,verbosity=-1)
    classifier = lgb.LGBMClassifier(**options)
    classifier.fit(x,target,sample_weight=np.where(target,7,1))
    good = iou >= 0.22
    ratios = np.tile(np.maximum(3,boxes[:,2:]-boxes[:,:2]),(1,2))
    regression = []
    for k in range(4):
        regressor = lgb.LGBMRegressor(**options)
        delta = ((near[:,k]-boxes[:,k])/ratios[:,k]).clip(-0.5,0.5)
        regressor.fit(x[good],delta[good],sample_weight=np.sqrt(iou[good]))
        regression.append(regressor.booster_)
    return classifier.booster_, regression


def score(model, sample, refined):
    classifier, regression = model
    boxes,x = sample
    probability = classifier.predict(x,num_threads=5)
    if refined:
        movement = np.column_stack([g.predict(x,num_threads=4) for g in regression]).clip(-0.45,0.45)
        ratios = np.tile(np.maximum(3,boxes[:,2:]-boxes[:,:2]),(1,2))
        boxes = boxes + 0.80*movement*ratios
        boxes = boxes.clip(0,[660,720,660,720])
    valid = (boxes[:,2:] > boxes[:,:2]).all(axis=1)
    return boxes[valid].astype(np.float32),probability[valid].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg",default="ffmpeg")
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT/"prospective")
    parser.add_argument("--include-full-frame", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    gray = load_gray(output,args.ffmpeg)
    _, ids_val, ids_dev = PROFILES["prospective"]
    val,_,_ = load_labels(output,ids_val,gray)
    dev,_,_ = load_labels(output,ids_dev,gray)
    configs = choose_sources(output, args.include_full_frame)
    features = feature_cache(output,gray,max(dev),CausalFeatures())
    raw_val, samples = make_records([load_candidates(output,cfg,"validation") for cfg in configs],val,features)
    raw_dev, _ = make_records([load_candidates(output,cfg,"development") for cfg in configs],dev,features)
    print("Candidate sources:",[model_key(c) for c in configs],flush=True)
    folds = np.array_split(np.array(sorted(raw_val)),3)
    out_of_fold = {False:{},True:{}}
    for fold in folds:
        held = set(fold)
        supervision = [i for i in val if i not in held]
        model = fit(samples,supervision)
        for mode in out_of_fold:
            for i in fold:
                out_of_fold[mode][i] = score(model,raw_val[i],mode)
    final_model = fit(samples,list(val))
    record = []
    for mode, calibration in out_of_fold.items():
        operation, metrics, _ = select_operation(calibration,val)
        candidates = with_nms({i:score(final_model,u,mode) for i,u in raw_dev.items()},operation["nms"])
        evaluation, _ = summarize(filtered(candidates,operation["threshold"],operation["low"]),dev)
        coverage, center = diagnostic_coverage(candidates,dev)
        row = {"refined":mode,"operation":operation,"validation":metrics,"development":evaluation,
               "development_recall_at_60":coverage,"development_center_coverage_at_60":center}
        record.append(row)
        print(row,flush=True)
    paths = []
    stem = "fusion-full-frame" if args.include_full_frame else "fusion"
    for index,model in enumerate((final_model[0],*final_model[1])):
        path = output/f"{stem}-{index}.txt"
        model.save_model(str(path))
        paths.append(path)
    write_json(output/f"{stem}-comparison.json",{"comparisons":record,"sources":configs,
               "estimators":{p.name:sha_file(p) for p in paths},
               "validation_method":"three chronological out-of-fold intervals"})


if __name__ == "__main__":
    main()
