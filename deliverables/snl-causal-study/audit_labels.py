"""Audit adjacent-frame LabelMe boxes against the SNL display image content."""
import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
from scipy.optimize import linear_sum_assignment

from snl_inputs import DEFAULT_OUTPUT, PROFILES, ROOT, blob_table, correlation, load_gray, load_labels, verified_bytes, write_json

sys.path.insert(0, str(ROOT))
from videosar.snl_causal import box_iou


def centers(boxes):
    return (boxes[:,:2]+boxes[:,2:])/2


def patch_motion(first,second,box):
    x0,y0,x1,y1=np.rint(box).astype(int)
    h,w=first.shape
    radius=7
    if x0<radius or y0<radius or x1+radius>w or y1+radius>h:
        return None
    sample=first[y0:y1,x0:x1].astype(np.float32)
    sample-=sample.mean()
    magnitude=np.linalg.norm(sample)
    if magnitude<1e-5:
        return None
    area=second[y0-radius:y1+radius,x0-radius:x1+radius].astype(np.float32)
    grid=sliding_window_view(area,sample.shape)
    grid=grid-grid.mean(axis=(-2,-1),keepdims=True)
    scores=np.einsum("yxij,ij->yx",grid,sample)/(np.linalg.norm(grid,axis=(-2,-1))*magnitude+1e-6)
    pos=np.unravel_index(scores.argmax(),scores.shape)
    return np.array([pos[1]-radius,pos[0]-radius]),float(scores[pos])


def analyze(gray, labels):
    displacement, overlaps, disagreements = [],[],[]
    local_correlations=[]
    large=0
    for i, old in labels.items():
        if i+1 not in labels:
            continue
        new=labels[i+1]
        predicted=centers(new)[None]-centers(old)[:,None]
        distance=np.linalg.norm(predicted,axis=-1)
        a,b=linear_sum_assignment(distance)
        iou=box_iou(old,new)
        for k,j in zip(a,b,strict=True):
            if distance[k,j]>15:
                large+=1
                continue
            displacement.append(float(distance[k,j]))
            overlaps.append(float(iou[k,j]))
            local=patch_motion(gray[i-2],gray[i-1],old[k])
            if local is not None:
                motion,corr=local
                if corr>=.80:
                    disagreements.append(float(np.linalg.norm(predicted[k,j]-motion)))
                    local_correlations.append(corr)
    return {
      "pairs_estimated":len(displacement),"pairs_displacement_over_15_px":large,
      "annotation_center_step_px_q50_q75_q90":np.percentile(displacement,[50,75,90]).tolist(),
      "annotation_iou_adjacent_q25_q50_q75":np.percentile(overlaps,[25,50,75]).tolist(),
      "pairs_with_image_patch_ncc_at_least_point8":len(disagreements),
      "disagreement_with_image_patch_motion_px_q50_q75_q90":np.percentile(disagreements,[50,75,90]).tolist(),
    }


def stabilize_training_boxes(gray, labels):
    """Blend each annotation with its preceding image-aligned training rectangle."""
    output = {}
    shifts = []
    for index, boxes in labels.items():
        revised = boxes.copy()
        if index - 1 in output:
            previous = labels[index - 1]
            distance = np.linalg.norm(centers(previous)[:, None] - centers(boxes)[None], axis=2)
            source, target = linear_sum_assignment(distance)
            for i, j in zip(source, target, strict=True):
                if distance[i, j] > 10:
                    continue
                match = patch_motion(gray[index - 3], gray[index - 2], previous[i])
                if match is None or match[1] < 0.88:
                    continue
                offset = np.tile(match[0], 2)
                predicted = output[index - 1][i] + offset
                if box_iou(predicted[None], boxes[j, None])[0, 0] < 0.20:
                    continue
                result = (0.62 * boxes[j] + 0.38 * predicted).clip(0, [660, 720, 660, 720])
                if np.all(result[2:] > result[:2]):
                    revised[j] = result
                    shifts.append(float(np.linalg.norm(centers(result[None])[0] - centers(boxes[j, None])[0])))
        output[index] = revised
    return output, {"frames": len(output), "original_boxes": sum(len(b) for b in labels.values()),
                    "stabilized_boxes": len(shifts),
                    "median_center_adjustment_px": float(np.median(shifts)) if shifts else None}


def alignment_in_labeled_neighborhoods(gray, labels, sources):
    blobs=blob_table()
    correlations=[]
    for i, boxes in list(labels.items())[::3]:
        if i<=70 or not len(boxes):
            continue
        raw=verified_bytes(sources,f"{i:06}",blobs[f"{i:06}"])
        original=json.loads(raw.decode("utf-8-sig"))
        image=Image.open(io.BytesIO(base64.b64decode(original["imageData"]))).convert("L")
        ref=np.asarray(image)
        support=np.zeros((704,644),dtype=bool)
        label_shape=boxes*[644/660,704/720,644/660,704/720]
        for x0,y0,x1,y1 in np.rint(label_shape).astype(int):
            support[max(0,y0-3):min(704,y1+3),max(0,x0-3):min(644,x1+3)]=True
        offset=[]
        for n in range(i-4,i+1):
            resized=np.asarray(Image.fromarray(gray[n]).resize((644,704),Image.Resampling.BILINEAR))
            offset.append(correlation(ref[support],resized[support]))
        correlations.append((i,offset))
    gaps=[max(row)-row[2] for _,row in correlations]
    return {"rows":correlations,"count":len(gaps),"shifted_better_at_least_0_01":int(np.sum(np.asarray(gaps)>=.01)),
            "mapped_target_crop_correlation_median":float(np.median([r[2] for _,r in correlations]))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg",default="ffmpeg")
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT/"prospective")
    args=parser.parse_args()
    root=args.output.resolve()
    gray=load_gray(root,args.ffmpeg)
    train,val,dev=[load_labels(root,ids,gray)[0] for ids in PROFILES["prospective"]]
    measurements={name:analyze(gray,label) for name,label in (("train",train),("validation",val),("development",dev))}
    for k,v in measurements.items():
        print(k,v,flush=True)
    for k,labels in (("validation",val),("development",dev)):
        local=alignment_in_labeled_neighborhoods(gray,labels,root)
        measurements[k]["local_alignment"]=local
        print(k,{key:value for key,value in local.items() if key!="rows"},flush=True)
    write_json(root/"label-continuity-audit.json",measurements)


if __name__=="__main__":
    main()
