"""Fit the native-grid center model with whole-frame scene context."""
import numpy as np
import torch

from snl_center import ARCHITECTURE, CenterDetector, center_loss
from snl_detector import backend
from snl_inputs import sha_file


def augmented_frame(image, boxes, rng):
    h,w=image.shape[-2:]
    pad=36
    dx,dy=rng.integers(-pad,pad+1,size=2)
    broad=np.pad(image,((0,0),(pad,pad),(pad,pad)),mode="edge")
    frame=broad[:,pad-dy:pad-dy+h,pad-dx:pad-dx+w]
    truth=boxes+np.array([dx,dy,dx,dy],dtype=np.float32)
    bounded=truth.clip(0,[w,h,w,h]).astype(np.float32)
    initial_area=(truth[:,2:]-truth[:,:2]).prod(axis=1)
    retained=(bounded[:,2:]-bounded[:,:2]).prod(axis=1)
    truth=bounded[(retained>=0.6*initial_area) & (np.min(bounded[:,2:]-bounded[:,:2],axis=1)>=2)]
    if rng.random()<0.5:
        frame=frame[...,::-1]
        truth[:,[0,2]]=w-truth[:,[2,0]]
    return frame.copy(),{"boxes":torch.from_numpy(truth),
                         "labels":torch.ones(len(truth),dtype=torch.int64)}


def train_full(output,features,labels):
    torch.manual_seed(6182)
    rng=np.random.default_rng(6182)
    channels=features.shape[1]
    device=backend()
    mixed=device.type=="cuda"
    model=CenterDetector(dense_offsets=True,input_channels=channels,global_context=True).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.00055,weight_decay=.0002)
    total=32
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,total,eta_min=.000055)
    scaler=torch.amp.GradScaler("cuda",enabled=mixed)
    ids=tuple(sorted(labels))
    for epoch in range(1,total+1):
        order=np.array(ids[epoch%2::2],dtype=int)
        rng.shuffle(order)
        losses=[]
        model.train()
        for start in range(0,len(order),2):
            batch=[augmented_frame(features[i-2],labels[i],rng) for i in order[start:start+2]]
            x=torch.from_numpy(np.stack([b[0] for b in batch])).to(device).float()/255
            shifts=torch.from_numpy(rng.uniform(.80,1.20,len(batch)).astype(np.float32)).to(device)
            x=(x*shifts[:,None,None,None]).clamp(0,1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16,enabled=mixed):
                logits=model(x)
            loss=center_loss(model,logits.float(),[b[1] for b in batch],soft_heat=True)
            if not torch.isfinite(loss):
                raise RuntimeError("Full-frame center detector produced a non-finite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),10)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        print(f"radar-full-context epoch {epoch}/{total}: loss {np.mean(losses):.4f}",flush=True)
        if epoch%8==0:
            path=output/f"radar-center-full-context-epoch{epoch}.pt"
            torch.save(model.state_dict(),path)
            yield {"architecture":ARCHITECTURE,"method":"temporal","scale":1,"epoch":epoch,
                   "path":path.name,"sha256":sha_file(path),"input_profile":"registered",
                   "global_context":True,"dense_offsets":True,"input_channels":channels,
                   "soft_label_heatmap":True,"training_label_frames":len(labels),
                   "device":device.type,"torch":str(torch.__version__)}
