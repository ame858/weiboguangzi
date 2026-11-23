"""High-resolution, multi-object center detector trained on native SNL rectangles."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from snl_detector import TILE, backend, detector_batch, draw_windows, tile_origins, training_tiles
from snl_inputs import sha_file

ARCHITECTURE = "radar-center"
STRIDE = 2
EPOCHS = 48


def convolution(c_in, c_out, stride=1):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
                         nn.GroupNorm(8, c_out), nn.SiLU())


def downsample(c_in, c_out, space_to_depth):
    if space_to_depth:
        return nn.Sequential(nn.PixelUnshuffle(2), convolution(4*c_in, c_out))
    return convolution(c_in, c_out, 2)


class TemporalContrast(nn.Module):
    """Use a shared spatial encoder before comparing the current view with preceding views."""

    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(nn.Conv2d(1, 8, 5, padding=2, bias=False), nn.SiLU())
        self.fuse = nn.Sequential(nn.Conv2d(32, 8, 1), nn.GroupNorm(4, 8), nn.SiLU())

    def forward(self, frames):
        batch, count, height, width = frames.shape
        levels = self.encode(frames.reshape(batch*count, 1, height, width))
        levels = levels.reshape(batch, count, -1, height, width)
        difference = levels[:, 1:] - levels[:, :1]
        cues = (difference[:, 0], difference[:, 1:4].mean(dim=1),
                difference.amax(dim=1), difference.amin(dim=1))
        return self.fuse(torch.cat(cues, dim=1))


class RecurrentTemporalContrast(nn.Module):
    """Carry spatial evidence from the oldest view to the current frame."""

    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(convolution(2, 8, 2), convolution(8, 8, 2))
        self.gates = nn.Conv2d(16, 16, 3, padding=1)
        self.candidate = nn.Conv2d(16, 8, 3, padding=1)

    def forward(self, frames):
        batch, count, height, width = frames.shape
        raw = frames.reshape(batch * count, 1, height, width)
        background = F.avg_pool2d(raw, 13, stride=1, padding=6, count_include_pad=False)
        encoded = self.encode(torch.cat((raw, 4 * (background - raw)), dim=1))
        levels = encoded.unflatten(0, (batch, count))
        state = torch.zeros_like(levels[:, 0])
        for index in reversed(range(count)):
            step = levels[:, index]
            reset, update = self.gates(torch.cat((step, state), dim=1)).sigmoid().chunk(2, dim=1)
            candidate = self.candidate(torch.cat((step, reset * state), dim=1)).tanh()
            state = torch.lerp(state, candidate, update)
        return F.interpolate(state, size=(height, width), mode="bilinear", align_corners=False)


class MixedTemporalContrast(nn.Module):
    """Encode ordered frame differences separately from the current-frame spatial branch."""

    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv3d(2, 8, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
            nn.GroupNorm(4, 8), nn.SiLU(),
            nn.Conv3d(8, 8, 3, stride=(1, 2, 2), padding=1, bias=False),
            nn.GroupNorm(4, 8), nn.SiLU(),
        )
        self.merge = nn.Conv2d(16, 8, 1)

    def forward(self, frames):
        ordered = frames.flip(1)
        # Signed differences retain both darkening and brightening, which distinguish moving shadows from static clutter.
        difference = torch.diff(ordered, dim=1, prepend=ordered[:, :1])
        relative = ordered - ordered.mean(dim=1, keepdim=True)
        encoded = self.encode(4 * torch.stack((difference, relative), dim=1))
        features = self.merge(torch.cat((encoded[:, :, -1], encoded.amax(dim=2)), dim=1))
        return F.interpolate(features, size=frames.shape[-2:], mode="bilinear", align_corners=False)


class LineContext(nn.Module):
    """Learn how image-axis streaks at stride four relate to the shadow center."""

    def __init__(self):
        super().__init__()
        self.reduce = nn.Conv2d(48, 8, 1, bias=False)
        self.horizontal = nn.Conv2d(8, 8, (1, 15), padding=(0, 7), groups=8, bias=False)
        self.vertical = nn.Conv2d(8, 8, (15, 1), padding=(7, 0), groups=8, bias=False)
        self.merge = nn.Conv2d(16, 48, 1, bias=False)
        nn.init.constant_(self.horizontal.weight, 1 / 15)
        nn.init.constant_(self.vertical.weight, 1 / 15)
        nn.init.zeros_(self.merge.weight)

    def forward(self, medium):
        signals = self.reduce(medium)
        local = F.avg_pool2d(signals, 3, stride=1, padding=1, count_include_pad=False)
        lines = torch.cat((self.horizontal(signals) - local, self.vertical(signals) - local), dim=1)
        return medium + self.merge(lines)


class CenterDetector(nn.Module):
    def __init__(self, dense_offsets=False, input_channels=3, global_context=False, learned_background=False,
                 space_to_depth=False, shared_temporal=False, line_context=False, recurrent_temporal=False,
                 mixed_temporal=False, task_alignment=False):
        super().__init__()
        self.dense_offsets = dense_offsets
        self.task_alignment = task_alignment
        self.background = None
        self.temporal_encoder = None
        self.mixed_temporal = mixed_temporal
        temporal_choices = (shared_temporal, recurrent_temporal, mixed_temporal)
        if any(temporal_choices):
            if input_channels != 7 or sum(temporal_choices) != 1:
                raise ValueError("A temporal experiment requires seven channels and one encoder")
            encoders = (TemporalContrast, RecurrentTemporalContrast, MixedTemporalContrast)
            self.temporal_encoder = encoders[temporal_choices.index(True)]()
        if mixed_temporal:
            if learned_background:
                raise ValueError("Mixed temporal features use the current image as their spatial input")
            input_channels = 1
        if learned_background:
            from snl_background import BackgroundPredictor

            if input_channels != 7:
                raise ValueError("The background predictor requires seven temporal channels")
            self.background = BackgroundPredictor().requires_grad_(False)
            input_channels += 2
        if self.temporal_encoder is not None:
            input_channels += 8
        self.space_to_depth = space_to_depth
        self.fine = nn.Sequential(convolution(input_channels+1, 24), downsample(24, 32, space_to_depth),
                                  convolution(32, 32))
        self.medium = nn.Sequential(downsample(32, 48, space_to_depth), convolution(48, 48))
        self.line_context = LineContext() if line_context else None
        self.coarse = nn.Sequential(downsample(48, 64, space_to_depth), convolution(64, 64))
        self.context = nn.MultiheadAttention(64, 4, batch_first=True) if global_context else None
        if global_context:
            self.context_norm = nn.LayerNorm(64)
        self.fuse_medium = convolution(112, 48)
        self.fuse_fine = nn.Sequential(convolution(80, 48), convolution(48, 32))
        self.head = nn.Conv2d(32, 6, 1)
        with torch.no_grad():
            self.head.bias[0] = -4.0
            self.head.bias[3:5] = torch.tensor([2.0, 2.6])

    def forward(self, features):
        if self.space_to_depth:
            height, width = features.shape[-2:]
            features = F.pad(features, (0, -width % 8, 0, -height % 8), mode="replicate")
        if self.background is not None:
            features = self.background.detector_channels(features)
        if self.temporal_encoder is not None:
            temporal = self.temporal_encoder(features[:, :7])
            spatial = features[:, :1] if self.mixed_temporal else features
            features = torch.cat((spatial, temporal), dim=1)
        gray = features[:, :1]
        inner = F.avg_pool2d(gray, 3, 1, padding=1, count_include_pad=False)
        outer = F.avg_pool2d(gray, 11, 1, padding=5, count_include_pad=False)
        fine = self.fine(torch.cat((features, 4 * (outer-inner)), dim=1))
        medium = self.medium(fine)
        if self.line_context is not None:
            medium = self.line_context(medium)
        coarse = self.coarse(medium)
        if self.context is not None:
            rows, columns = 12, 12
            pooled = F.adaptive_avg_pool2d(coarse, (rows, columns))
            tokens = pooled.flatten(2).transpose(1, 2)
            normalized = self.context_norm(tokens)
            global_features = self.context(normalized, normalized, normalized, need_weights=False)[0]
            grid = global_features.transpose(1, 2).reshape(len(features), 64, rows, columns)
            coarse = coarse + F.interpolate(grid, size=coarse.shape[-2:], mode="bilinear", align_corners=False)
        up = F.interpolate(coarse, size=medium.shape[-2:], mode="bilinear", align_corners=False)
        mid = self.fuse_medium(torch.cat((medium, up), dim=1))
        up = F.interpolate(mid, size=fine.shape[-2:], mode="bilinear", align_corners=False)
        output = self.head(self.fuse_fine(torch.cat((fine, up), dim=1)))
        if self.space_to_depth:
            output = output[..., :(height+1)//2, :(width+1)//2]
        return output

    def predict_boxes(self, frame):
        return decode_frame(self, frame)

    def center_offsets(self, logits):
        if self.task_alignment:
            return 3.0 * logits.tanh()
        return 1.5 * logits.tanh() if self.dense_offsets else logits.sigmoid()


class CenterMapEnsemble(nn.Module):
    def __init__(self, members, input_channels):
        super().__init__()
        self.models = nn.ModuleList(CenterDetector(input_channels=input_channels) for _ in range(members))

    def forward(self, frames):
        return fuse_center_maps([model(frames) for model in self.models])

    def predict_boxes(self, frame):
        return decode_frame(self, frame)

    def center_offsets(self, logits):
        return logits.sigmoid()


def fuse_center_maps(outputs):
    individual = torch.stack(outputs)
    probability = individual[:, :, :1].sigmoid()
    pooled = probability.mean(dim=0).clamp(1e-6, 1-1e-6)
    weights = probability.square()
    geometry = (individual[:, :, 1:] * weights).sum(dim=0) / weights.sum(dim=0).clamp_min(1e-9)
    return torch.cat((torch.logit(pooled), geometry), dim=1)


def restore_horizontal_map(flipped):
    restored = flipped.flip(-1)
    # With stride 2 and an even frame width, the reflected in-cell offset is 1 - sigmoid(logit).
    restored[:, 1].neg_()
    return restored


def local_gain_view(tensor, gain):
    local = F.avg_pool2d(tensor, 11, stride=1, padding=5, count_include_pad=False)
    return torch.lerp(local, tensor, gain).clamp(0, 1)


def predict_views(model, tensor):
    views = [model(tensor)]
    if getattr(model, "flip_inference", False):
        if tensor.shape[-1] % STRIDE or getattr(model, "dense_offsets", False):
            raise ValueError("Horizontal map fusion requires an even frame width and bounded center offsets")
        views.append(restore_horizontal_map(model(tensor.flip(-1))))
    if gain := getattr(model, "contrast_gain", 0):
        views.append(model(local_gain_view(tensor, gain)))
    return fuse_center_maps(views) if len(views) > 1 else views[0]


def stitched_maps(model, tensor):
    height, width = tensor.shape[-2:]
    if min(height, width) < TILE:
        return predict_views(model, tensor)
    tensor = F.pad(tensor, (0, width % STRIDE, 0, height % STRIDE), mode="replicate")
    height, width = tensor.shape[-2:]
    origins = tile_origins(height, width)
    size = TILE // STRIDE
    axis = torch.arange(size, device=tensor.device, dtype=tensor.dtype)
    taper = (torch.minimum(axis + 1, size - axis) / 12).clamp(0.1, 1)
    window = taper[:, None] * taper[None]
    shape = (height // STRIDE, width // STRIDE)
    scores, weights, geometry_weights = (tensor.new_zeros(shape) for _ in range(3))
    geometry = tensor.new_zeros((5, *shape))
    for start in range(0, len(origins), 4):
        group = origins[start:start+4]
        tiles = torch.cat([tensor[..., top:top+TILE, left:left+TILE] for left, top in group])
        for (left, top), output in zip(group, predict_views(model, tiles), strict=True):
            y, x = top // STRIDE, left // STRIDE
            region = np.s_[y:y+size, x:x+size]
            probability = output[0].sigmoid()
            vote = window * probability.square()
            weights[region] += window
            scores[region] += window * probability
            geometry_weights[region] += vote
            geometry[:, y:y+size, x:x+size] += output[1:] * vote
    likelihood = torch.logit((scores / weights).clamp(1e-6, 1 - 1e-6))
    return torch.cat((likelihood[None], geometry / geometry_weights.clamp_min(1e-9)), dim=0)[None]


def decode_frame(model, frame):
    device = next(model.parameters()).device
    tensor = torch.from_numpy(frame.copy()).to(device).float()[None] / 255
    h, w = frame.shape[-2:]
    with torch.inference_mode():
        predictor = stitched_maps if getattr(model, "stitch_inference", False) else predict_views
        output = predictor(model, tensor)[0]
        return decode_center_map(model, output, h, w)


def decode_center_map(model, output, h, w):
    heat = output[0].sigmoid()
    if power := getattr(model, "occupancy_power", 0):
        heat = heat * output[5].sigmoid().pow(power)
    dense_limit = getattr(model, "dense_candidate_limit", 0)
    if not dense_limit:
        local = heat == F.max_pool2d(heat[None, None], 3, 1, 1)[0, 0]
        heat = heat.masked_fill(~local, 0)
    score = heat.flatten()
    strength, index = score.topk(min(dense_limit or 256, len(score)))
    valid = strength >= 0.005
    strength, index = strength[valid], index[valid]
    boxes = center_boxes_at(model, output, index, h, w)
    valid = (boxes[:, 2:] > boxes[:, :2]).all(dim=1)
    return boxes[valid].cpu().numpy(), strength[valid].cpu().numpy()


def center_boxes_at(model, output, index, h, w):
    columns = output.shape[-1]
    y, x = torch.div(index, columns, rounding_mode="floor"), index % columns
    delta = model.center_offsets(output[1:3, y, x].T)
    center = STRIDE * (torch.stack((x, y), dim=1) + delta)
    half = output[3:5, y, x].T.clamp(0, 4.6).exp() / 2
    boxes = torch.cat((center - half, center + half), dim=1)
    boxes[:, 0::2].clamp_(0, w)
    boxes[:, 1::2].clamp_(0, h)
    return boxes


def make_targets(targets, height, width, dense_offsets=False, soft_heat=False):
    heat = np.zeros((len(targets), height, width), dtype=np.float32)
    foreground = np.zeros_like(heat)
    addresses, values = [], []
    for i, entry in enumerate(targets):
        boxes = entry["boxes"].numpy()
        for x0, y0, x1, y1 in boxes:
            cx, cy = (x0 + x1) / (2 * STRIDE), (y0 + y1) / (2 * STRIDE)
            x, y = (int(np.rint(cx)), int(np.rint(cy))) if dense_offsets else (int(cx), int(cy))
            x,y = min(x,width-1),min(y,height-1)
            if not (0 <= x < width and 0 <= y < height):
                continue
            sigma = np.clip(min(x1-x0, y1-y0) / (3 * STRIDE), 1, 2.5)
            if soft_heat:
                sigma = np.sqrt(sigma**2 + 1.35**2)
            radius = int(np.ceil(3 * sigma))
            l, r, b, t = max(0, x-radius), min(width, x+radius+1), max(0, y-radius), min(height, y+radius+1)
            dx, dy = np.arange(l, r) - x, np.arange(b, t) - y
            bump = np.exp(-(dx[None, :]**2 + dy[:, None]**2) / (2 * sigma**2))
            np.maximum(heat[i, b:t, l:r], bump, out=heat[i, b:t, l:r])
            uncertainty = 1 if dense_offsets else 0
            for yy in range(max(0,y-uncertainty),min(height,y+uncertainty+1)):
                for xx in range(max(0,x-uncertainty),min(width,x+uncertainty+1)):
                    addresses.append((i, yy, xx))
                    values.append((cx-xx, cy-yy, np.log(x1-x0), np.log(y1-y0)))
            foreground[i, int(y0//STRIDE):int(np.ceil(y1/STRIDE)),
                       int(x0//STRIDE):int(np.ceil(x1/STRIDE))] = 1
    return heat, foreground, np.asarray(addresses, dtype=np.int64).reshape(-1, 3), np.asarray(values, dtype=np.float32).reshape(-1, 4)


def box_projection_loss(logits, foreground, targets, current):
    from torchvision.ops import roi_align

    probability = logits.sigmoid()
    inside = foreground.bool()
    background = (F.softplus(logits) * ~inside).sum() / (~inside).sum().clamp_min(1)
    rectangles = [item["boxes"].to(logits) / STRIDE for item in targets]
    patches = roi_align(probability[:, None], rectangles, (8, 8), aligned=True)
    projection = logits.new_zeros(())
    if len(patches):
        across, along = patches.amax(dim=-1), patches.amax(dim=-2)
        projection = -0.5 * (across.clamp_min(1e-6).log().mean() + along.clamp_min(1e-6).log().mean())
    gray = F.avg_pool2d(current, STRIDE, stride=STRIDE, ceil_mode=True)
    consistency, total = logits.new_zeros(()), logits.new_zeros(())
    for dimension in (-2, -1):
        count = probability.shape[dimension] - 1
        first = probability.narrow(dimension, 0, count)
        second = probability.narrow(dimension, 1, count)
        nearby = inside.narrow(dimension, 0, count) | inside.narrow(dimension, 1, count)
        difference = (gray.narrow(dimension, 0, count) - gray.narrow(dimension, 1, count)).abs()
        weight = torch.exp(-difference / 0.025) * nearby
        agreement = first * second + (1 - first) * (1 - second)
        consistency -= (weight * agreement.clamp_min(1e-6).log()).sum()
        total += weight.sum()
    return background + projection + 0.15 * consistency / total.clamp_min(1)


def peak_ranking_loss(logits, heat):
    """Rank annotated centers ahead of high-scoring background peaks in the same tile."""
    with torch.no_grad():
        peaks = logits == F.max_pool2d(logits[:, None], 3, stride=1, padding=1)[:, 0]
        negatives = peaks & (heat < 0.05)
    top = logits.masked_fill(~negatives, -1e4).flatten(1).topk(min(16, logits[0].numel())).values
    batch, y, x = torch.where(heat == 1)
    positive = logits[batch, y, x]
    margin = F.softplus(1.0 + top[batch] - positive[:, None])
    return margin.sum() / max(1, margin.numel())


def nwd_similarity(center_a, size_a, center_b, size_b, scale):
    delta = torch.cat((center_a - center_b, (size_a - size_b) / 2), dim=1)
    return torch.exp(-torch.linalg.vector_norm(delta, dim=1) / scale)


def center_loss(model, output, targets, soft_heat=False, quality_focal=False, mask_images=None, rank_peaks=False,
                nwd_scale=None):
    device = output.device
    heat, foreground, addresses, values = (torch.from_numpy(value).to(device)
                                          for value in make_targets(targets, *output.shape[-2:],
                                                                     model.dense_offsets, soft_heat))
    probability = output[:, 0].sigmoid()
    centers = heat == 1
    if mask_images is None:
        occupancy = F.binary_cross_entropy_with_logits(output[:, 5], foreground, pos_weight=output.new_tensor(4))
    else:
        occupancy = box_projection_loss(output[:, 5], foreground, targets, mask_images)
    if model.task_alignment:
        from snl_task_alignment import task_alignment_loss

        return task_alignment_loss(model, output, targets, STRIDE) + 0.25 * occupancy
    i, y, x = addresses.T
    regression = output[i, 1:5, y, x]
    center = model.center_offsets(regression[:, :2])
    offset = F.smooth_l1_loss(center, values[:, :2], reduction="sum")
    size = F.smooth_l1_loss(regression[:, 2:], values[:, 2:], reduction="sum")
    geometry_loss = (2 * offset + size) / max(1, len(addresses))
    overlap_quality = None
    if model.dense_offsets and len(addresses):
        p_center, t_center = STRIDE * center, STRIDE * values[:, :2]
        p_size, t_size = regression[:, 2:].clamp(0,4.6).exp(), values[:, 2:].exp()
        p_low, p_high = p_center - p_size/2, p_center + p_size/2
        t_low, t_high = t_center - t_size/2, t_center + t_size/2
        inter = (torch.minimum(p_high,t_high)-torch.maximum(p_low,t_low)).clamp_min(0).prod(dim=1)
        union = p_size.prod(dim=1) + t_size.prod(dim=1) - inter
        encompassing = (torch.maximum(p_high,t_high)-torch.minimum(p_low,t_low)).prod(dim=1)
        overlap_quality = inter / union
        if nwd_scale is None:
            geometry_loss += (1 - overlap_quality + (encompassing-union)/encompassing).mean()
        else:
            overlap_quality = nwd_similarity(p_center, p_size, t_center, t_size, nwd_scale)
            geometry_loss += (1 - overlap_quality).mean()
    if quality_focal:
        flat_quality = probability.new_zeros((probability.numel(),))
        if overlap_quality is not None:
            height, width = probability.shape[-2:]
            location = (i * height + y) * width + x
            flat_quality.scatter_reduce_(0, location, overlap_quality.detach().to(flat_quality.dtype), reduce="amax")
        quality = flat_quality.view_as(probability)
        bce = F.binary_cross_entropy_with_logits(output[:, 0], quality, reduction="none")
        class_loss = (bce * (probability - quality).square()).sum() / centers.sum().clamp_min(1)
    else:
        positive = -(1 - probability).square() * F.logsigmoid(output[:, 0])
        negative = -(1 - heat).pow(4) * probability.square() * F.logsigmoid(-output[:, 0])
        if soft_heat:
            soft_positive = F.binary_cross_entropy_with_logits(output[:, 0], heat, reduction="none") * heat
            class_loss = (soft_positive + negative).sum() / heat.sum().clamp_min(1)
        else:
            class_loss = torch.where(centers, positive, negative).sum() / centers.sum().clamp_min(1)
    if rank_peaks:
        class_loss += 0.5 * peak_ranking_loss(output[:, 0], heat)
    return class_loss + geometry_loss + 0.25 * occupancy


def train_center(output, features, labels, *, augment=False, motion_bank=None, input_profile="raw", dense_offsets=False,
                 initial=None, stabilized_labels=False, soft_heat=False, learned_background=False,
                 space_to_depth=False, shared_temporal=False, line_context=False, weak_shadow_augment=False,
                 quality_focal=False, rebalance_targets=False, box_projection=False, local_gain_augment=False,
                 seed=1729, pretrained=None, pretraining_epochs=None, recurrent_temporal=False,
                 mixed_temporal=False, rank_peaks=False, task_alignment=False, hard_tiles=(),
                 nwd_scale=None, epoch_budget=None):
    if initial and pretrained:
        raise ValueError("Choose one initialization checkpoint")
    if nwd_scale is not None and (not np.isfinite(nwd_scale) or nwd_scale <= 0 or not dense_offsets or task_alignment):
        raise ValueError("NWD needs a positive finite scale and the dense geometry head")
    if epoch_budget is not None and (epoch_budget < 12 or epoch_budget % 12 or pretraining_epochs is not None or initial):
        raise ValueError("The training budget must be a positive multiple of the normal checkpoint interval")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = backend()
    channels = features.shape[1]
    background = None
    if learned_background:
        from snl_background import train_background

        background = train_background(output, features, labels, device, seed)
        torch.manual_seed(seed)
    model = CenterDetector(dense_offsets=dense_offsets, input_channels=channels,
                           learned_background=learned_background, space_to_depth=space_to_depth,
                           shared_temporal=shared_temporal, line_context=line_context,
                           recurrent_temporal=recurrent_temporal, mixed_temporal=mixed_temporal,
                           task_alignment=task_alignment).to(device)
    if background is not None:
        model.background.load_state_dict(background.state_dict())
    epochs, learning_rate, checkpoint_interval = epoch_budget or EPOCHS, 0.001, 12
    if pretraining_epochs is not None:
        if pretraining_epochs < 1 or pretrained or initial:
            raise ValueError("Source pretraining needs a positive budget and fresh initialization")
        epochs = checkpoint_interval = pretraining_epochs
    if pretrained:
        path = output / pretrained["path"]
        if sha_file(path) != pretrained["sha256"]:
            raise ValueError("Synthetic pretraining checkpoint checksum mismatch")
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    if initial:
        path = output / initial["path"]
        if sha_file(path) != initial["sha256"]:
            raise ValueError("Recent fine-tune source has a different checksum")
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        cutoff = max(labels) - 94
        labels = {index: boxes for index, boxes in labels.items() if index >= cutoff or index % 7 == 0}
        epochs, learning_rate, checkpoint_interval = 24, 0.00025, 6
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.1*learning_rate)
    ids, (_, _, height, width) = tuple(sorted(labels)), features.shape
    tile_transform = None
    object_weights = {}
    sampling = None
    if weak_shadow_augment:
        from snl_augmentation import weak_shadows

        tile_transform = weak_shadows(labels, seed)
    if rebalance_targets:
        from snl_augmentation import weak_target_weights

        object_weights, sampling = weak_target_weights(features, labels)
    if augment:
        from snl_augmentation import copy_paste, training_chips
        chips = training_chips(features, labels, margin=12 if input_profile == "stacked" else 5)
    tag = ARCHITECTURE + ("-augmented" if augment else "")
    if input_profile != "raw":
        tag += f"-{input_profile}"
    if dense_offsets:
        tag += "-iou"
    if nwd_scale is not None:
        tag += "-nwd"
    if initial:
        tag += "-recent"
    if pretraining_epochs is not None:
        tag += "-simulation-source"
    if pretrained:
        tag += "-simulation-transfer"
    if stabilized_labels:
        tag += "-stable-labels"
    if soft_heat:
        tag += "-soft-labels"
    if space_to_depth:
        tag += "-space-depth"
    if shared_temporal:
        tag += "-shared-temporal"
    if recurrent_temporal:
        tag += "-recurrent-temporal"
    if mixed_temporal:
        tag += "-mixed-temporal"
    if line_context:
        tag += "-line-context"
    if weak_shadow_augment:
        tag += "-weak-shadows"
    if quality_focal:
        tag += "-quality-focal"
    if rank_peaks:
        tag += "-peak-ranking"
    if task_alignment:
        tag += "-task-aligned"
    if rebalance_targets:
        tag += "-weak-sampling"
    if box_projection:
        tag += "-box-projection"
    if local_gain_augment:
        tag += "-local-gain"
        gain_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    if motion_bank:
        from snl_motion_aug import synthetic_batch

        tag += "-motion"
    if seed != 1729:
        tag += f"-seed{seed}"
    for epoch in range(1, epochs + 1):
        records = training_tiles(ids, labels, width, height, rng, epoch, hard_tiles, object_weights=object_weights)
        for i in ids:
            records.append((i, *draw_windows(labels[i], width, height, rng, object_weights.get(i))))
        rng.shuffle(records)
        model.train()
        losses = []
        for start in range(0, len(records), 12):
            batch, targets = detector_batch(records[start:start+12], features, labels, "temporal", rng,
                                            tile_transform=tile_transform)
            if augment:
                copy_paste(batch, targets, chips, rng, stacked=input_profile == "stacked")
            if motion_bank and len(batch) >= 4:
                amount = len(batch) // 3
                examples, boxes = synthetic_batch(motion_bank, amount, rng)
                batch[-amount:] = examples
                targets[-amount:] = boxes
            images = torch.stack(batch).to(device)
            # Brightness changes are applied consistently within each input map.
            levels = torch.from_numpy(rng.uniform(0.85, 1.15, len(batch)).astype(np.float32)).to(device)
            images = (images * levels[:, None, None, None]).clamp_(0, 1)
            if local_gain_augment:
                gains = gain_rng.uniform(0.60, 1.50, len(batch)).astype(np.float32)
                gains[gain_rng.random(len(batch)) < 0.25] = 1.0
                images = local_gain_view(images, torch.from_numpy(gains).to(device)[:, None, None, None])
            optimizer.zero_grad(set_to_none=True)
            mask_images = images[:, 0] if box_projection else None
            loss = center_loss(model, model(images), targets, soft_heat=soft_heat,
                               quality_focal=quality_focal, mask_images=mask_images, rank_peaks=rank_peaks,
                               nwd_scale=nwd_scale)
            if not torch.isfinite(loss):
                raise RuntimeError("Center detector produced a non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        print(f"{tag} epoch {epoch}/{epochs}: loss {np.mean(losses):.4f}", flush=True)
        if epoch % checkpoint_interval == 0:
            path = output / f"{tag}-epoch{epoch}.pt"
            torch.save(model.state_dict(), path)
            config = {"architecture": ARCHITECTURE, "method": "temporal", "scale": 1, "epoch": epoch, "seed": seed,
                   "path": path.name, "sha256": sha_file(path), "device": device.type,
                   "torch": str(torch.__version__), "training_copy_paste": augment,
                   "training_motion_synthesis": bool(motion_bank), "input_profile": input_profile,
                   "dense_offsets": dense_offsets, "recent_fine_tune": bool(initial),
                   "initial_sha256": initial["sha256"] if initial else None,
                   "training_label_frames": len(labels), "stabilized_training_labels": stabilized_labels,
                   "soft_label_heatmap": soft_heat, "input_channels": channels}
            if pretrained:
                config["pretrained_sha256"] = pretrained["sha256"]
            if hard_tiles:
                config["extra_training_tiles"] = len(hard_tiles)
            if nwd_scale is not None:
                config["nwd_scale_px"] = nwd_scale
            if learned_background:
                config["learned_background"] = True
            if space_to_depth:
                config["space_to_depth"] = True
            if shared_temporal:
                config["shared_temporal"] = True
            if recurrent_temporal:
                config["recurrent_temporal"] = True
            if mixed_temporal:
                config["mixed_temporal"] = True
            if line_context:
                config["line_context"] = True
            if weak_shadow_augment:
                config["weak_shadow_augmentation"] = True
            if quality_focal:
                config["quality_focal_training"] = True
            if rank_peaks:
                config["training_peak_ranking"] = True
            if task_alignment:
                config["task_alignment"] = True
            if sampling is not None:
                config["weak_target_sampling"] = sampling
            if box_projection:
                config["box_projection_supervision"] = True
            if local_gain_augment:
                config["local_gain_augmentation"] = [0.60, 1.50]
            yield config
