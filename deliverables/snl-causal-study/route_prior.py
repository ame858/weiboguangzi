"""Evaluate a road-corridor prior learned from the early labeled trajectories."""
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from experiment import CONFIDENCES, filtered, model_key, quality, summarize, with_nms
from snl_inputs import DEFAULT_OUTPUT, HEIGHT, PROFILES, WIDTH, load_gray, load_labels, read_json, write_json


def centers(boxes):
    return (boxes[:,:2]+boxes[:,2:])/2


def paths(labels):
    tracks = []
    ongoing = {}
    previous = np.empty((0,2),dtype=np.float32)
    last = None
    for i, boxes in labels.items():
        found = centers(boxes)
        if last is None or i-last != 1:
            ongoing = {}
            previous = previous[:0]
        distance = np.linalg.norm(previous[:,None]-found[None],axis=2)
        r,c = linear_sum_assignment(distance)
        mapping = {}
        for a,b in zip(r,c,strict=True):
            if distance[a,b] <= 4:
                j = ongoing[a]
                tracks[j].append(found[b])
                mapping[b] = j
        for k,value in enumerate(found):
            if k not in mapping:
                mapping[k] = len(tracks)
                tracks.append([value])
        previous,ongoing,last = found,mapping,i
    return [np.stack(t) for t in tracks]


def raster_line(mask, points):
    for p,q in zip(points[:-1],points[1:],strict=True):
        count = max(2, int(np.linalg.norm(q-p)*2))
        coords = np.rint(np.linspace(p,q,count)).astype(int)
        x,y = coords.T
        valid = (x>=0)&(x<WIDTH)&(y>=0)&(y<HEIGHT)
        mask[y[valid],x[valid]] = True


def corridor(labels, length):
    base = np.zeros((HEIGHT,WIDTH),bool)
    for boxes in labels.values():
        point = np.rint(centers(boxes)).astype(int).clip(0, [WIDTH-1, HEIGHT-1])
        x,y = point.T
        base[y,x] = True
    for move in paths(labels):
        if len(move) < 14:
            continue
        raster_line(base,move)
        for edge in (move,move[::-1]):
            end = edge[-1]
            local = edge[-min(30,len(edge)):]
            t = np.arange(len(local),dtype=np.float32)
            vel = ((t-t.mean())[:,None]*(local-local.mean(axis=0))).sum(axis=0)/((t-t.mean())**2).sum()
            speed = np.linalg.norm(vel)
            if speed >= 0.25:
                raster_line(base,[end,end+length*vel/speed])
    return ndimage.distance_transform_edt(~base).astype(np.float32)


def candidate_data(output, config, scope):
    raw = read_json(output/f"raw-{model_key(config)}-{scope}.json")
    return {int(i):(np.asarray(v[0],dtype=np.float32).reshape(-1,4),np.asarray(v[1],dtype=np.float32))
            for i,v in raw["predictions"].items()}


def apply_prior(candidates, distance, radius, strength):
    values = {}
    for index,(boxes,scores) in candidates.items():
        loc = np.rint(centers(boxes)).astype(int).clip(0,[WIDTH-1,HEIGHT-1])
        delta = distance[loc[:,1],loc[:,0]]
        if strength == 0:
            mask = delta <= radius
            values[index] = boxes[mask],scores[mask]
        else:
            values[index] = boxes,scores*np.exp(-0.5*(delta/radius)**2*strength)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg",default="ffmpeg")
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT/"prospective")
    args = parser.parse_args()
    root = args.output.resolve()
    gray = load_gray(root,args.ffmpeg)
    labels = [load_labels(root,i,gray)[0] for i in PROFILES["prospective"]]
    train,val,dev = labels
    corridors = {v:corridor(train,v) for v in (0,45,90)}
    oracle = {}
    for length, map_ in corridors.items():
        coord = np.rint(np.concatenate([centers(x) for x in dev.values()])).astype(int)
        distance = map_[coord[:,1],coord[:,0]]
        oracle[length] = {r:float(np.mean(distance<=r)) for r in (10,20,35,55,75,100)}
    print("Annotated center coverage by corridor:",oracle,flush=True)
    configs = [read_json(p) for p in root.glob("decision-*.json")]
    chosen = sorted((x for x in configs if "path" in x["config"]),
                    key=lambda v:v["development"]["f1"],reverse=True)[:5]
    outcome = []
    for trial in chosen:
        config = trial["config"]
        valid = with_nms(candidate_data(root,config,"validation"),config["nms"])
        development = with_nms(candidate_data(root,config,"development"),config["nms"])
        grid = []
        for length, dist in corridors.items():
            for radius in (15,25,40,60,85):
                for strength in (0,1):
                    records = apply_prior(valid,dist,radius,strength)
                    for threshold in CONFIDENCES:
                        value,_ = summarize(filtered(records,float(threshold),None),val)
                        grid.append((quality(value),length,radius,strength,float(threshold),value))
        _, length,radius,strength,threshold, calibration = max(grid,key=lambda x:x[0])
        predict = apply_prior(development,corridors[length],radius,strength)
        validation,_ = summarize(filtered(predict,threshold,None),dev)
        record = {"model":model_key(config),"config":config,"extension_px":length,"radius_px":radius,
                  "strength":strength,"threshold":threshold,"validation":calibration,"development":validation}
        outcome.append(record)
        print(record,flush=True)
    write_json(root/"road-corridor-comparison.json",{"coverage":oracle,"comparisons":outcome})


if __name__ == "__main__":
    main()
