"""Forecast static SAR background from older images using target-masked supervision."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from snl_inputs import write_json


def smooth(image):
    return F.avg_pool2d(image, 3, stride=1, padding=1, count_include_pad=False)


class BackgroundPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = nn.Sequential(nn.Conv2d(6, 16, 3, padding=1), nn.SiLU(),
                                   nn.Conv2d(16, 16, 3, padding=2, dilation=2), nn.SiLU())
        self.mix = nn.Sequential(nn.Conv2d(16, 16, 3, padding=4, dilation=4), nn.SiLU(),
                                 nn.Conv2d(16, 7, 3, padding=1))
        nn.init.zeros_(self.mix[-1].weight)
        nn.init.zeros_(self.mix[-1].bias)

    def forward(self, past):
        history = smooth(past)
        alternatives = self.mix(self.local(history))
        weights = alternatives[:, :6].softmax(dim=1)
        prediction = (weights * history).sum(dim=1, keepdim=True)
        return (prediction + 0.15 * alternatives[:, 6:].tanh()).clamp(0, 1)

    def detector_channels(self, stack):
        current, past = stack[:, :1], stack[:, 1:]
        observed = smooth(current)
        residual = self(past) - observed
        # The first frame repeats the current image in all unavailable history slots.
        no_history = (past == current).flatten(1).all(dim=1).reshape(-1, 1, 1, 1)
        residual = residual.masked_fill(no_history, 0)
        return torch.cat((stack, (4 * residual).clamp(0, 1), (-4 * residual).clamp(0, 1)), dim=1)


def background_mask(shape, boxes):
    mask = np.ones(shape, dtype=bool)
    height, width = shape
    for x0, y0, x1, y1 in boxes:
        left = max(0, int(np.floor(x0)) - 3)
        top = max(0, int(np.floor(y0)) - 3)
        right = min(width, int(np.ceil(x1)) + 3)
        bottom = min(height, int(np.ceil(y1)) + 3)
        mask[top:bottom, left:right] = False
    return mask


def train_background(output, features, labels, device, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = BackgroundPredictor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    size = 96
    height, width = features.shape[-2:]
    ids = [index for index in sorted(labels) if index >= 32]
    masks = {index: background_mask((height, width), labels[index]) for index in ids}
    records = [(index, top, left) for index in ids
               for top in range(0, height - size + 1, size)
               for left in range(0, width - size + 1, size)]
    history = []
    for epoch in range(1, 7):
        # Rotate the tile coverage each epoch without mixing in a later frame.
        order = rng.permutation(len(records))[:4 * len(ids)]
        errors = []
        for start in range(0, len(order), 24):
            inputs, valid = [], []
            for position in order[start:start+24]:
                index, top, left = records[position]
                region = np.s_[top:top+size, left:left+size]
                inputs.append(features[index - 2, :, top:top+size, left:left+size])
                valid.append(masks[index][region])
            stack = torch.from_numpy(np.stack(inputs)).to(device, dtype=torch.float32) / 255
            keep = torch.from_numpy(np.stack(valid)).to(device)[:, None]
            keep = keep[:, :, 8:-8, 8:-8]
            observed = smooth(stack[:, :1])[:, :, 8:-8, 8:-8]
            predicted = model(stack[:, 1:])[:, :, 8:-8, 8:-8]
            loss = ((predicted - observed).abs() * keep).sum() / keep.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            errors.append(float(loss.detach()))
        history.append(float(np.mean(errors)))
        print(f"Causal background epoch {epoch}/6: masked MAE {history[-1]:.5f}", flush=True)
    write_json(output / f"background-training-seed{seed}.json", {"seed": seed, "frame_ids": ids,
               "lags": [1, 3, 6, 12, 24, 30], "excluded_margin_px": 3, "masked_training_mae": history})
    return model.requires_grad_(False)
