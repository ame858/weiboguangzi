"""A causal dense SNL detector using supervised local radar statistics."""
import argparse
from collections import deque
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter

from experiment import filtered, propose, select_operation, summarize, diagnostic_coverage, with_nms
from snl_inputs import (
    DEFAULT_OUTPUT, DEVELOPMENT, TRAIN, VALIDATE, load_gray, load_labels, read_json, sha_file, write_json,
)

METHOD = "dense-radar-statistics"
STRIDE = 2
LAGS = (6, 12, 18, 24, 30, 45, 60, 75, 90)


def mean(image, width, height):
    return uniform_filter(image, size=(height, width), mode="nearest")


def shifted(image, dx, dy):
    yy = np.clip(np.arange(image.shape[0]) + dy, 0, image.shape[0] - 1)
    xx = np.clip(np.arange(image.shape[1]) + dx, 0, image.shape[1] - 1)
    return image[yy[:, None], xx[None, :]]


class CausalStatistics:
    def __init__(self):
        self.history = deque(maxlen=max(LAGS))

    def update(self, frame, *, used=True):
        gray = np.asarray(frame, dtype=np.float32) / 255
        smooth = mean(gray, 3, 3)
        result = self.features(gray, smooth) if used else None
        self.history.append(smooth)
        return result

    def features(self, gray, smooth):
        history = {lag: self.history[-lag] for lag in LAGS if len(self.history) >= lag}
        fallback = np.zeros_like(gray)
        stack = np.stack(list(history.values())) if history else smooth[None]
        median = np.median(stack, axis=0)
        dispersion = np.median(np.abs(stack-median), axis=0)
        upper = np.quantile(stack, .75, axis=0).astype(np.float32)
        temporal = np.maximum(0, upper-smooth)
        delta6 = history[6]-smooth if 6 in history else fallback
        delta24 = history[24]-smooth if 24 in history else fallback
        images = [gray, smooth, temporal, temporal/(.02+dispersion), delta6, delta24]
        for w,h in ((5,9),(7,13),(9,17),(13,23)):
            interior = mean(gray,w,h)
            outside = mean(gray,w+12,h+12)
            contrast = outside-interior
            horizontal = w//2+4
            vertical = h//2+4
            left,right = (shifted(mean(gray,5,h),d,0)-interior for d in (-horizontal,horizontal))
            top,bottom = (shifted(mean(gray,w,5),0,d)-interior for d in (-vertical,vertical))
            noise = np.sqrt(np.clip(mean(gray**2,w,h)-interior*interior,0,1))
            images.extend((interior,contrast,contrast/(.02+outside),left,right,top,bottom,noise,
                           mean(temporal,w,h),mean(delta6,w,h),mean(delta24,w,h)))
        return np.stack(images, axis=-1)


def training_records(features, boxes, rng):
    h,w = features.shape[:2]
    centers = (boxes[:,:2]+boxes[:,2:])/2
    x, y, offsets, sizes = [],[],[],[]
    spread = np.array([(dx,dy) for dx in (-1,0,1) for dy in (-2,-1,0,1,2)])
    for box, center in zip(boxes, centers, strict=True):
        proposed = np.rint(center).astype(int) + spread
        proposed = proposed[((proposed>=0)&(proposed<[w,h])).all(axis=1)]
        x.extend(proposed[:,0])
        y.extend(proposed[:,1])
        offsets.extend(center-proposed)
        sizes.extend([np.log(box[2:]-box[:2])]*len(proposed))
    positive = features[np.asarray(y,dtype=int),np.asarray(x,dtype=int)]
    rgb = features[..., 6] if features.shape[-1]>6 else features[..., 0]
    proxy = -rgb + features[..., 7] + 0.35*features[..., 2]
    peak = proxy == maximum_filter(proxy,size=5,mode="nearest")
    hard_y,hard_x = np.nonzero(peak)
    order = np.argsort(-proxy[hard_y,hard_x],kind="stable")[:450]
    hard = np.column_stack((hard_x[order],hard_y[order]))
    uniform = rng.integers([0,0],[w,h],size=(320,2))
    mixed = np.concatenate((hard,uniform))
    inside = ((mixed[:,None]>=boxes[None,:,:2]-3)&(mixed[:,None]<=boxes[None,:,2:]+3)).all(axis=-1)
    keep = ~inside.any(axis=1)
    negative = mixed[keep]
    false = features[negative[:,1],negative[:,0]]
    return positive,false,np.asarray(offsets,dtype=np.float32).reshape(-1,2),np.asarray(sizes,dtype=np.float32).reshape(-1,2)


def fit(training, labels, output):
    rng = np.random.default_rng(719)
    stats = CausalStatistics()
    arrays = [[],[],[],[]]
    used_ids = tuple(labels)[::2]
    chosen = set(used_ids)
    for i in range(2,max(chosen)+1):
        features = stats.update(training[i-2],used=i in chosen)
        if features is None:
            continue
        for store,value in zip(arrays,training_records(features,labels[i],rng),strict=True):
            store.append(value)
        if i % 40 == 2:
            print(f"Dense training measurements through frame {i}",flush=True)
    positive,negative,delta,geometry = map(np.concatenate,arrays)
    data = np.concatenate((positive,negative))
    truth = np.concatenate((np.ones(len(positive)),np.zeros(len(negative))))
    options = dict(n_estimators=235,num_leaves=31,max_depth=7,learning_rate=0.06,n_jobs=8,
                   min_child_samples=24,random_state=719,verbosity=-1)
    foreground = lgb.LGBMClassifier(**options)
    foreground.fit(data,truth,sample_weight=np.concatenate((np.full(len(positive),3.),np.ones(len(negative)))))
    paths = []
    for model,name in ((foreground,"classification"),):
        path = output/f"dense-{name}.txt"
        model.booster_.save_model(path)
        paths.append(path)
    for j,target in enumerate(np.column_stack((delta,geometry)).T):
        regressor = lgb.LGBMRegressor(**options)
        regressor.fit(positive,target)
        path = output / f"dense-regression-{j}.txt"
        regressor.booster_.save_model(path)
        paths.append(path)
    config = {"method":METHOD,"paths":{p.name:sha_file(p) for p in paths},
              "training_frames":list(used_ids),"positives":len(positive),"negatives":len(negative)}
    write_json(output/"dense-fit.json",config)
    return config


class DenseDetector:
    def __init__(self,output,config):
        self.features = CausalStatistics()
        self.models = {}
        for name,digest in config["paths"].items():
            path = output/name
            if sha_file(path) != digest:
                raise ValueError(f"{name}: model digest mismatch")
            self.models[name] = lgb.Booster(model_file=str(path))

    def update(self, frame, *, used=True):
        features = self.features.update(frame,used=used)
        if features is None:
            return None
        h,w = features.shape[:2]
        grid = features[::STRIDE,::STRIDE]
        ny,nx = grid.shape[:2]
        columns = grid.reshape(ny*nx,-1)
        likelihood = self.models["dense-classification.txt"].predict(columns,num_threads=8).reshape(ny,nx)
        peak = likelihood == maximum_filter(likelihood,size=3,mode="nearest")
        y,x = np.nonzero(peak)
        order = np.argsort(-likelihood[y,x],kind="stable")[:320]
        y,x = y[order],x[order]
        features = grid[y,x]
        refinements = np.column_stack([self.models[f"dense-regression-{i}.txt"].predict(features,num_threads=4) for i in range(4)])
        center = np.column_stack((x,y))*STRIDE+refinements[:,:2].clip(-2,2)
        half = np.exp(refinements[:,2:].clip(np.log(3),np.log(48)))/2
        boxes = np.column_stack((center-half,center+half)).clip(0,[w,h,w,h]).astype(np.float32)
        good = np.all(boxes[:,2:]>boxes[:,:2],axis=1)
        return boxes[good],likelihood[y,x][good].astype(np.float32)


def predictions(output,config,gray,targets):
    detector = DenseDetector(output,config)
    ids = {i for n in targets for i in range(max(2,n-6),n+1)}
    captured = {}
    for i in range(2,max(ids)+1):
        item = detector.update(gray[i-2],used=i in ids)
        if item is not None:
            captured[i] = item
    return captured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg",default="ffmpeg")
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT/"gpu")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    gray = load_gray(output,args.ffmpeg)
    fit_path = output/"dense-fit.json"
    if fit_path.is_file():
        config = read_json(fit_path)
    else:
        train_labels,_,_ = load_labels(output,TRAIN,gray)
        config = fit(gray,train_labels,output)
    validation,_,_ = load_labels(output,VALIDATE,gray)
    development,_,_ = load_labels(output,DEVELOPMENT,gray)
    all_predictions = predictions(output,config,gray,validation | development)
    evaluation = {i:all_predictions[i] for n in validation for i in range(max(2,n-6),n+1)}
    thresholds, val, _ = select_operation(evaluation,validation)
    candidates = with_nms(all_predictions,thresholds["nms"])
    dev,_ = summarize(filtered(candidates,thresholds["threshold"],thresholds["low"]),development)
    coverage, centers = diagnostic_coverage(candidates,development)
    decision = {"config":config|thresholds,"validation":val,"development":dev,
                "development_recall_at_60":coverage,"development_center_coverage_at_60":centers}
    write_json(output/"decision-dense-statistics.json",decision)
    print(decision,flush=True)
    propose(output)


if __name__ == "__main__":
    main()
