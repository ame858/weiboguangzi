"""Train a shadow candidate classifier on early real crops and detector false alarms."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiment import diagnostic_coverage, filtered, model_key, raw_candidates, select_operation, summarize, with_nms
from snl_detector import backend, nonmax
from snl_inputs import DEFAULT_OUTPUT, PROFILES, feature_cache, load_gray, load_labels, read_json, sha_file, write_json
from videosar.snl_causal import CausalFeatures, box_iou

SIDELENGTH = 40
SOURCES = 6
SOURCE_CAP = 240
POOL_CAP = 850


def source_records(output):
    selections = {}
    for path in output.glob("decision-*.json"):
        proposal = read_json(path)
        cfg = proposal["config"]
        if "path" not in cfg:
            continue
        key = tuple(cfg.get(s) for s in ("architecture","method","input_profile","training_motion_synthesis",
                                         "training_copy_paste","dense_offsets","recent_fine_tune"))
        if key not in selections or proposal["validation"]["f1"] > selections[key]["validation"]["f1"]:
            selections[key] = proposal
    return [r["config"] for r in sorted(selections.values(),key=lambda r:r["validation"]["f1"],reverse=True)[:SOURCES]]


def proposals(estimators,index,*,source_cap=SOURCE_CAP,pool_cap=POOL_CAP):
    boxes,scores=[],[]
    for estimator in estimators:
        locations,quality=estimator[index]
        order=np.argsort(-quality,kind="stable")[:source_cap]
        boxes.append(locations[order])
        scores.append(quality[order])
    b,p=nonmax(np.concatenate(boxes),np.concatenate(scores),0.68)
    return b[:pool_cap],p[:pool_cap]


def patch_inputs(frame,boxes):
    h,w = frame.shape[-2:]
    n=len(boxes)
    center=np.rint(0.5*(boxes[:,:2]+boxes[:,2:])).astype(int)
    offsets=np.arange(SIDELENGTH)-SIDELENGTH//2
    x=center[:,0,None]+offsets
    y=center[:,1,None]+offsets
    sample=frame[:,y[:,:,None].clip(0,h-1),x[:,None,:].clip(0,w-1)].transpose((1,0,2,3))
    inside_x=(x>=boxes[:,0,None])&(x<boxes[:,2,None])
    inside_y=(y>=boxes[:,1,None])&(y<boxes[:,3,None])
    prior=inside_y[:,:,None]&inside_x[:,None,:]
    result=np.empty((n,4,SIDELENGTH,SIDELENGTH),dtype=np.uint8)
    result[:,:3]=sample
    result[:,3]=prior.astype(np.uint8)*255
    return result


def random_boxes(truth,rng):
    jitter=np.repeat(truth,8,axis=0)
    half=(jitter[:,2:]-jitter[:,:2])/2
    ctr=(jitter[:,:2]+jitter[:,2:])/2
    ctr+=rng.uniform(-0.70,0.70,size=ctr.shape)*half
    half*=rng.uniform(0.70,1.35,size=half.shape)
    return np.concatenate((truth,np.column_stack((ctr-half,ctr+half))),axis=0)


def collect_training(output, configs, features, truth):
    ids=tuple(truth)[::4]
    estimators=[raw_candidates(output,cfg,features[cfg.get("input_profile","raw")],ids,"patch-train")
                for cfg in configs[:2]]
    return training_patches(estimators,features["registered"],truth,ids)


def training_patches(estimators, features, truth, ids, *, source_cap=SOURCE_CAP, pool_cap=POOL_CAP):
    rng=np.random.default_rng(127)
    images,classes,geometry=[],[],[]
    for i in ids:
        targets=truth[i]
        if not len(targets):
            continue
        boxes,_=proposals(estimators,i,source_cap=source_cap,pool_cap=pool_cap)
        iou=box_iou(boxes,targets)
        high=iou.max(axis=1)
        hard=np.flatnonzero(high<.24)
        hard=hard[:min(64,len(hard))]
        real=np.flatnonzero(high>=.24)
        real=real[:min(80,len(real))]
        mixed=np.concatenate((random_boxes(targets,rng),boxes[hard],boxes[real]))
        mixed=mixed.clip(0,[660,720,660,720]).astype(np.float32)
        valid=(mixed[:,2:]-mixed[:,:2]>1).all(axis=1)
        mixed=mixed[valid]
        overlap=box_iou(mixed,targets)
        covered=overlap.max(axis=1)>=0.30
        assigned=targets[overlap.argmax(axis=1)]
        size=np.tile(np.maximum(3,mixed[:,2:]-mixed[:,:2]),(1,2))
        delta=((assigned-mixed)/size).clip(-0.75,.75)
        images.append(patch_inputs(features[i-2],mixed))
        classes.append(covered.astype(np.float32))
        geometry.append(delta.astype(np.float32))
    return tuple(np.concatenate(x) for x in (images,classes,geometry))


class PatchScore(nn.Module):
    def __init__(self, outputs=5):
        super().__init__()
        layers=[]
        ins=4
        for outs in (24,48,64):
            layers.extend((nn.Conv2d(ins,outs,3,padding=1,bias=False),nn.GroupNorm(8,outs),
                           nn.SiLU(),nn.MaxPool2d(2)))
            ins=outs
        self.features=nn.Sequential(*layers)
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d(3),nn.Flatten(),nn.Linear(64*9,128),nn.SiLU(),nn.Linear(128,outputs))

    def forward(self, x):
        return self.head(self.features(x))


def train(output, records):
    torch.manual_seed(127)
    rng=np.random.default_rng(127)
    device=backend()
    model=PatchScore().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0002)
    x,y,delta=records
    for epoch in range(1,21):
        order=rng.permutation(len(y))
        model.train()
        losses=[]
        for start in range(0,len(order),96):
            keys=order[start:start+96]
            roi=torch.from_numpy(x[keys]).float().to(device)/255
            truth=torch.from_numpy(y[keys]).to(device)
            regression=torch.from_numpy(delta[keys]).to(device)
            rate=torch.from_numpy(rng.uniform(.8,1.2,len(keys)).astype(np.float32)).to(device)
            roi[:,:3]=(roi[:,:3]*rate[:,None,None,None]).clamp(0,1)
            predicted=model(roi)
            geo=F.smooth_l1_loss(predicted[:,1:].tanh()*0.75,regression,reduction="none").mean(axis=1)
            loss=F.binary_cross_entropy_with_logits(predicted[:,0],truth)+(geo*truth).sum()/truth.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch % 5 == 0:
            print(f"Patch scoring epoch {epoch}/20: loss {np.mean(losses):.4f}",flush=True)
    path=output/"patch-score.pt"
    torch.save(model.state_dict(),path)
    return model.eval(),{"path":path.name,"sha256":sha_file(path)}


def evaluate_neural(model,estimators,features,ids,*,source_cap=SOURCE_CAP,pool_cap=POOL_CAP):
    raw={}
    device=next(model.parameters()).device
    with torch.inference_mode():
        for i in ids:
            boxes,first=proposals(estimators,i,source_cap=source_cap,pool_cap=pool_cap)
            patches=patch_inputs(features[i-2],boxes)
            if len(patches):
                output=torch.cat([model(chunk.float().to(device)/255)
                                  for chunk in torch.from_numpy(patches).split(160)]).cpu()
                probability=output[:,0].sigmoid().numpy().astype(np.float32)
                move=(output[:,1:].tanh()*0.75).numpy()
                size=np.tile(np.maximum(3,boxes[:,2:]-boxes[:,:2]),(1,2))
                corrected=(boxes+move*size).clip(0,[660,720,660,720]).astype(np.float32)
            else:
                probability=first.copy()
                corrected=boxes.copy()
            raw[i]=boxes,corrected,probability,first
    return raw


def sources(output,configs,scope):
    result=[]
    for cfg in configs:
        payload=read_json(output/f"raw-{model_key(cfg)}-{scope}.json")
        if payload["checkpoint_sha256"]!=cfg["sha256"]:
            raise ValueError("Cached detector checksum changed")
        result.append({int(i):(np.asarray(b,dtype=np.float32).reshape(-1,4),np.asarray(s,dtype=np.float32))
                       for i,(b,s) in payload["predictions"].items()})
    return result


def choose(raw,share,refine):
    data={}
    for i,(boxes,fixed,cnn,other) in raw.items():
        selected=fixed if refine else boxes
        valid=(selected[:,2:]>selected[:,:2]).all(axis=-1)
        scores=((1-share)*cnn+share*other)[valid]
        data[i]=selected[valid],scores
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg",default="ffmpeg")
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT/"prospective")
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    out=args.output.resolve()
    gray=load_gray(out,args.ffmpeg)
    training,val,dev=[load_labels(out,ids,gray)[0] for ids in PROFILES["prospective"]]
    configs=source_records(out)
    maps={name:feature_cache(out,gray,max(dev),CausalFeatures(name))
          for name in {"registered",*(c.get("input_profile","raw") for c in configs[:2])}}
    data=collect_training(out,configs,maps,training)
    print(f"Supervised patches: {len(data[1])} total, {int(data[1].sum())} positives.",flush=True)
    model,checkpoint=train(out,data)
    val_sources=sources(out,configs,"validation")
    dev_sources=sources(out,configs,"development")
    val_output=evaluate_neural(model,val_sources,maps["registered"],sorted(val_sources[0]))
    dev_output=evaluate_neural(model,dev_sources,maps["registered"],sorted(dev_sources[0]))
    trial=[]
    for share in (0,.25,.50,.75):
        for refine in (False,True):
            op,v,_=select_operation(choose(val_output,share,refine),val)
            candidates=with_nms(choose(dev_output,share,refine),op["nms"])
            d,_=summarize(filtered(candidates,op["threshold"],op["low"]),dev)
            coverage,centers=diagnostic_coverage(candidates,dev)
            record={"source_score_weight":share,"refined":refine,"operation":op,"validation":v,"development":d,
                    "development_recall_at_60":coverage,"development_center_coverage_at_60":centers}
            trial.append(record)
            print(record,flush=True)
    picked=max(trial,key=lambda r:r["validation"]["f1"] or 0)
    write_json(out/"patch-score-comparison.json",{"model":checkpoint,"sources":configs,"selected":picked,"trials":trial})


if __name__=="__main__":
    main()
