"""Generate physically coherent causal augmentation on real background crops."""
import numpy as np
import torch
from scipy.ndimage import uniform_filter
from scipy.optimize import linear_sum_assignment

from snl_detector import TILE
from videosar.snl_causal import STACKED_HISTORY_LAGS

LAGS = (0, 6, 12, 18, 24, 30)


def local_darkness(frame, box):
    x0, y0, x1, y1 = np.rint(box).astype(int)
    height, width = frame.shape
    x0, y0 = np.clip((x0, y0), 0, (width - 1, height - 1))
    x1, y1 = np.clip((x1, y1), (x0 + 1, y0 + 1), (width, height))
    left, top = max(0, x0 - 4), max(0, y0 - 4)
    right, bottom = min(frame.shape[1], x1 + 4), min(frame.shape[0], y1 + 4)
    region = frame[top:bottom, left:right]
    rim = np.ones(region.shape, dtype=bool)
    rim[y0-top:y1-top, x0-left:x1-left] = False
    if not rim.any():
        return 0.0
    return float(np.median(region[rim]) - np.median(frame[y0:y1, x0:x1]))


def measurements(gray, labels):
    shapes, contrasts, displacement = [], [], []
    for index, boxes in labels.items():
        frame = np.asarray(gray[index-2], dtype=np.float32) / 255
        for box in boxes:
            x0, y0, x1, y1 = box
            contrast = local_darkness(frame, box)
            shapes.append((x1-x0,y1-y0))
            contrasts.append(max(0.008,contrast))
        previous = labels.get(index-6)
        if previous is None:
            continue
        orig = (previous[:,:2]+previous[:,2:])/2
        current = (boxes[:,:2]+boxes[:,2:])/2
        deltas = current[:,None]-orig[None]
        distance = np.linalg.norm(deltas,axis=-1)
        rows,cols = linear_sum_assignment(distance)
        displacement.extend((deltas[rows,cols][distance[rows,cols] < 12] / 6).tolist())
    return np.asarray(shapes),np.asarray(contrasts),np.asarray(displacement)


def background_tile(gray, labels, rng, eligible, lags):
    _, height, width = gray.shape
    for _ in range(200):
        i = int(rng.choice(eligible))
        left, top = rng.integers([0,0],[width-TILE+1,height-TILE+1])
        source = np.concatenate([labels[i-lag] for lag in lags])
        lo = np.array([left,top])
        hi = lo+TILE
        overlap = ((source[:,:2] < hi)&(source[:,2:] > lo)).all(axis=1)
        if not overlap.any():
            return np.stack([gray[i-lag-2,top:top+TILE,left:left+TILE] for lag in lags]).astype(np.float32)/255
    raise RuntimeError("Could not sample a target-free causal background")


def shape_on_tile(frame, center, half, theta, depth):
    radius = int(np.ceil(np.linalg.norm(half))+4)
    x,y = center
    left,right = max(0,int(x)-radius), min(TILE,int(x)+radius+1)
    top,bottom = max(0,int(y)-radius), min(TILE,int(y)+radius+1)
    dx = np.arange(left,right,dtype=np.float32)[None,:]-x
    dy = np.arange(top,bottom,dtype=np.float32)[:,None]-y
    c,s = np.cos(theta),np.sin(theta)
    u,v = c*dx+s*dy,-s*dx+c*dy
    outside = np.hypot(np.maximum(np.abs(u)-half[0],0),np.maximum(np.abs(v)-half[1],0))
    opacity = np.exp(-0.5*(outside/0.75)**2)
    patch = frame[top:bottom,left:right]
    level = max(float(np.median(patch)),0.03)
    attenuation = np.clip(depth/level,0.025,0.80)
    patch *= (1-attenuation*opacity).astype(np.float32)


def channels_from_observations(hist):
    smooth = uniform_filter(hist,size=(1,3,3),mode="nearest")
    short = np.maximum(0,smooth[1]-smooth[0])
    background = np.quantile(smooth[1:],0.75,axis=0).astype(np.float32)
    residual = np.maximum(0,background-smooth[0])
    return np.stack((hist[0],np.clip(4*short,0,1),np.clip(4*residual,0,1)))


def inject_shadows(background, priors, rng, lags):
    sizes, depths, velocity = priors
    truth = []
    for _ in range(int(rng.integers(1, 4))):
        key = rng.integers(len(sizes))
        half = 0.5 * sizes[key] * rng.uniform(0.82, 1.35, size=2)
        angle = rng.uniform(-0.30, 0.30)
        delta = velocity[rng.integers(len(velocity))] * rng.uniform(0.6, 1.7)
        past = np.array(lags)[:, None] * delta
        bound = int(np.linalg.norm(half) + 5)
        low = bound + np.maximum(0, past.max(axis=0))
        high = TILE - bound + np.minimum(0, past.min(axis=0))
        if np.any(low >= high):
            continue
        center = rng.uniform(low, high)
        projected = np.abs(np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])) @ half
        box = np.concatenate((center - projected, center + projected)).astype(np.float32)
        if any(np.all(np.minimum(box[2:], old[2:] + 3) > np.maximum(box[:2], old[:2] - 3)) for old in truth):
            continue
        depth = float(depths[key] * rng.uniform(0.45, 1.20))
        for frame, offset in zip(background, past, strict=True):
            shape_on_tile(frame, center - offset, half, angle, depth)
        truth.append(box)
    return np.asarray(truth, dtype=np.float32).reshape(-1, 4)


def generate_bank(gray, labels, *, count=1500, stacked=False):
    rng = np.random.default_rng(20261121)
    priors = measurements(gray, labels)
    lags = (0, *STACKED_HISTORY_LAGS) if stacked else LAGS
    eligible = [i for i in labels if all(i-lag in labels for lag in lags)]
    if not len(eligible) or not len(priors[2]):
        raise ValueError("Motion augmentation needs consecutive labeled training frames")
    bank = []
    for n in range(count):
        background = background_tile(gray, labels, rng, eligible, lags)
        boxes = inject_shadows(background, priors, rng, lags)
        channels = background if stacked else channels_from_observations(background)
        features = np.rint(255*channels).astype(np.uint8)
        bank.append((features,boxes))
        if (n+1) % 500 == 0:
            print(f"Generated {n+1}/{count} causal shadow clips",flush=True)
    return bank


def synthetic_batch(bank, size, rng):
    images,targets = [],[]
    for k in rng.integers(len(bank),size=size):
        image,boxes = bank[k]
        images.append(torch.from_numpy(image).float().div_(255))
        targets.append({"boxes":torch.from_numpy(boxes),"labels":torch.ones(len(boxes),dtype=torch.long)})
    return images,targets
