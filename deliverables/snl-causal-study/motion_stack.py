"""Forward-only motion integration of weak SNL shadow evidence."""
import argparse
from collections import Counter, deque
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from experiment import select_operation, filtered, with_nms, summarize, diagnostic_coverage
from snl_inputs import DEFAULT_OUTPUT, PROFILES, load_gray, load_labels, write_json
from videosar.snl_causal import CausalFeatures

STEPS = (6, 12, 18)
SHAPES = ((5,9),(7,13),(9,17),(11,21),(13,25))


def velocity_bank(labels):
    votes = Counter()
    for i, boxes in labels.items():
        if i-6 not in labels:
            continue
        past = labels[i-6]
        a = (boxes[:,:2]+boxes[:,2:])/2
        b = (past[:,:2]+past[:,2:])/2
        delta = a[:,None]-b[None,:]
        distance = np.linalg.norm(delta,axis=2)
        row,col = linear_sum_assignment(distance)
        for x,y in delta[row,col][distance[row,col] <= 10]:
            if np.hypot(x,y) >= 1.25:
                votes[(int(np.rint(x)),int(np.rint(y)))] += 1
    return [(0,0), *(v for v,_ in votes.most_common(15))]


def dark_score(channels):
    image = channels[0]
    context = ndimage.uniform_filter(image,size=(23,13),mode="nearest")
    interior = ndimage.uniform_filter(image,size=(9,5),mode="nearest")
    static = np.maximum(0,context-interior)/(0.045+context)
    moving = ndimage.uniform_filter(channels[2],size=(9,5),mode="nearest")
    return np.minimum(static + 0.55*moving,0.75)


def integrate(current, past, velocities):
    best = current.copy()
    for x,y in velocities:
        motion = current.copy()
        for step in STEPS:
            warp = ndimage.shift(past[-step],(y*step/6,x*step/6),order=1,mode="nearest",prefilter=False)
            motion += warp
        value = np.minimum(current * 1.35, motion / (1 + len(STEPS)))
        np.maximum(best,value,out=best)
    return best


def candidates(surface, gray):
    score = np.full(surface.shape,-np.inf,dtype=np.float32)
    size = np.zeros(surface.shape,dtype=np.int16)
    for j,(w,h) in enumerate(SHAPES):
        inner = ndimage.uniform_filter(gray,size=(h,w),mode="nearest")
        outer = ndimage.uniform_filter(gray,size=(h+8,w+8),mode="nearest")
        value = 0.7*surface + 2*(outer-inner)
        value[:h//2] = -np.inf
        value[-(h//2):] = -np.inf
        value[:,:w//2] = -np.inf
        value[:,-(w//2):] = -np.inf
        better = value>score
        score[better], size[better] = value[better],j
    y,x = np.nonzero((score==ndimage.maximum_filter(score,size=5))&np.isfinite(score))
    order = np.argsort(-score[y,x],kind="stable")[:384]
    y,x = y[order],x[order]
    w,h = np.array(SHAPES)[size[y,x]].T
    boxes = np.column_stack((x-w//2,y-h//2,x+w//2+1,y+h//2+1)).astype(np.float32)
    return boxes, score[y,x]


def raw_predictions(gray, indices, velocities):
    wanted = {i for j in indices for i in range(max(2,j-6),j+1)}
    history=deque(maxlen=max(STEPS))
    scanner=CausalFeatures("registered")
    output={}
    for i in range(2,max(wanted)+1):
        channels,_,_ = scanner.update(gray[i-2])
        current=dark_score(channels)
        if i in wanted:
            motion=integrate(current,history,velocities) if len(history)>=max(STEPS) else current
            output[i]=candidates(motion,channels[0])
        history.append(current)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg",default="ffmpeg")
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT/"prospective")
    args=parser.parse_args()
    out=args.output.resolve()
    frames=load_gray(out,args.ffmpeg)
    train,val,dev=[load_labels(out,ids,frames)[0] for ids in PROFILES["prospective"]]
    velocities=velocity_bank(train)
    raw=raw_predictions(frames,(*val,*dev),velocities)
    val_raw={i:raw[i] for n in val for i in range(max(2,n-6),n+1)}
    op,v,_=select_operation(val_raw,val)
    ordered=with_nms(raw,op["nms"])
    d,_=summarize(filtered(ordered,op["threshold"],op["low"]),dev)
    coverage,center=diagnostic_coverage(ordered,dev)
    record={"velocities_px_per_6_frames":velocities,"operating_point":op,"validation":v,"development":d,
            "development_recall_at_60":coverage,"development_center_coverage_at_60":center}
    write_json(out/"motion-stack-development.json",record)
    print(record,flush=True)


if __name__=="__main__":
    main()
