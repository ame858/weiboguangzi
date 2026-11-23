"""Causally rescore neural boxes by their short, motion-consistent history."""
import argparse
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from experiment import diagnostic_coverage, filtered, model_key, nonmax, select_operation, summarize, with_nms
from motion_stack import velocity_bank
from snl_inputs import DEFAULT_OUTPUT, PROFILES, load_gray, load_labels, read_json, write_json

LAGS = (1, 2, 3, 4, 6)


def centers(boxes):
    return (boxes[:,:2]+boxes[:,2:])/2


class MotionEvidence:
    def __init__(self,bank):
        self.history = deque(maxlen=max(LAGS))
        self.velocities = np.asarray(bank,dtype=np.float32)/6

    def calculate(self,index,boxes,scores):
        original = boxes
        if len(boxes):
            loc = centers(boxes)
            speed = self.velocities
            confidence = np.zeros((len(loc),len(speed)),dtype=np.float32)
            count = np.zeros_like(confidence)
            geometric = np.zeros((*confidence.shape,4),dtype=np.float32)
            for old,points,old_boxes,old_scores in self.history:
                delay = index-old
                if delay not in LAGS or len(old_boxes)==0:
                    continue
                motion = delay*speed
                target = loc[:,None,:]-motion[None,:,:]
                dist,which = points.query(target,k=1)
                shape = np.maximum(boxes[:,None,2:]-boxes[:,None,:2],1)
                history_shape = old_boxes[which,2:]-old_boxes[which,:2]
                shape_difference = np.max(np.abs(np.log(np.maximum(history_shape,1)/shape)),axis=-1)
                supported = (dist<=2.4)&(shape_difference<0.65)
                reliability = np.exp(-dist**2/5.5)*old_scores[which]*supported
                weights = np.concatenate((motion,motion),axis=1)
                aligned = old_boxes[which]+weights[None,:,:]
                geometric += aligned * reliability[...,None]
                confidence += reliability
                count += supported
            best = confidence.argmax(axis=1)
            r = np.arange(len(loc))
            prior = confidence[r,best] / len(LAGS)
            votes = count[r,best]
            reliability = confidence[r,best]
            movement = geometric[r,best]
            refined = np.where((votes>=2)[:,None],
                               (scores[:,None]*boxes+0.45*movement)/(scores[:,None]+0.45*reliability[:,None]+1e-9),
                               boxes)
            refined = refined.astype(np.float32)
        else:
            prior = scores.copy()
            refined = boxes.copy()
        self.history.append((index,cKDTree(centers(original)),original,scores))
        return refined,prior


def load_raw(out,config,scope):
    p=read_json(out/f"raw-{model_key(config)}-{scope}.json")
    if p["checkpoint_sha256"]!=config["sha256"]:
        raise ValueError("Candidate cache does not match the frozen model")
    return {int(i):(np.asarray(b,dtype=np.float32).reshape(-1,4),np.asarray(s,dtype=np.float32))
            for i,(b,s) in p["predictions"].items()}


def sequence(raw,velocities):
    tracker=MotionEvidence(velocities)
    results={}
    for i,(boxes,scores) in raw.items():
        boxes,scores=nonmax(boxes,scores,0.55)
        boxes,scores=boxes[:180],scores[:180]
        refined,prior=tracker.calculate(i,boxes,scores)
        results[i]=(boxes,refined,scores,prior)
    return results


def operation(values,weight,refine):
    return {i:(record[1] if refine else record[0],np.maximum(record[2],(1-weight)*record[2]+weight*record[3]*2.5))
            for i,record in values.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg",default="ffmpeg")
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT/"prospective")
    args=parser.parse_args()
    out=args.output.resolve()
    gray=load_gray(out,args.ffmpeg)
    train,val,dev=[load_labels(out,ids,gray)[0] for ids in PROFILES["prospective"]]
    velocities=velocity_bank(train)
    choices=[read_json(p) for p in out.glob("decision-*.json")]
    choices=sorted((r for r in choices if "path" in r["config"]),
                   key=lambda r:r["development"]["f1"],reverse=True)[:4]
    candidates=[]
    for row in choices:
        config=row["config"]
        validation=sequence(load_raw(out,config,"validation"),velocities)
        development=sequence(load_raw(out,config,"development"),velocities)
        options=[]
        for weight in (0,0.25,0.5,0.75):
            for refine in (False,True):
                point,summary,_=select_operation(operation(validation,weight,refine),val)
                options.append((summary["f1"],weight,refine,point,summary))
        _,weight,refine,point,calibration=max(options,key=lambda q:q[0])
        native=with_nms(operation(development,weight,refine),point["nms"])
        result,_=summarize(filtered(native,point["threshold"],point["low"]),dev)
        coverage,centers_at60=diagnostic_coverage(native,dev)
        trial={"source":config,"weight":weight,"refine":refine,"operating_point":point,
               "validation":calibration,"development":result,"development_recall_at_60":coverage,
               "development_center_coverage_at_60":centers_at60}
        candidates.append(trial)
        print({"source":model_key(config),"weight":weight,"refine":refine,"development":result},flush=True)
    write_json(out/"neural-motion-comparison.json",{"velocities":velocities,"comparisons":candidates})


if __name__=="__main__":
    main()
