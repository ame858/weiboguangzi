"""Joint box-quality supervision and adaptive geometry assignment for dense SNL candidates."""

import torch
from torch.nn import functional as F


def task_alignment_loss(model, output, targets, stride):
    logits = output[:, 0]
    batch_size, height, width = logits.shape
    boxes = torch.cat([target["boxes"] for target in targets]).to(output)
    quality = torch.zeros_like(logits)
    geometry = output.new_zeros(())
    if len(boxes):
        counts = torch.tensor([len(target["boxes"]) for target in targets], device=output.device)
        frames = torch.arange(batch_size, device=output.device).repeat_interleave(counts, output_size=len(boxes))
        center_cells = ((boxes[:, :2] + boxes[:, 2:]) / (2 * stride)).long()
        offsets = torch.arange(-2, 3, device=output.device)
        anchors = center_cells[:, None] + torch.cartesian_prod(offsets, offsets)
        x, y = anchors.unbind(dim=-1)
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x, y = x.clamp(0, width - 1), y.clamp(0, height - 1)
        frame_ids = frames[:, None]
        values = output[frame_ids, :, y, x]
        centers = stride * (anchors + model.center_offsets(values[..., 1:3]))
        half = values[..., 3:5].clamp(0, 4.6).exp() / 2
        lower, upper = centers - half, centers + half
        target_low, target_high = boxes[:, None, :2], boxes[:, None, 2:]
        overlap = (torch.minimum(upper, target_high) - torch.maximum(lower, target_low)).clamp_min(0).prod(dim=-1)
        union = (2 * half).prod(dim=-1) + (target_high - target_low).prod(dim=-1) - overlap
        extent = (torch.maximum(upper, target_high) - torch.minimum(lower, target_low)).prod(dim=-1)
        iou = overlap / union.clamp_min(1e-6)
        giou = iou - (extent - union) / extent.clamp_min(1e-6)
        locations = (frame_ids * height + y) * width + x
        with torch.no_grad():
            measured = iou.detach()
            quality.view(-1).scatter_reduce_(0, locations[valid], measured[valid], reduce="amax")
            alignment = values[..., 0].sigmoid().sqrt() * measured.pow(6)
            top = alignment.masked_fill(~valid, -1).topk(9, dim=1).indices
            chosen_locations = locations.gather(1, top)
            chosen_iou = measured.gather(1, top)
            chosen_valid = valid.gather(1, top)
            weights = chosen_iou.clamp_min(0.20) * chosen_valid
            # Where two small boxes share a grid cell, train geometry for the better overlap.
            weights *= chosen_iou >= quality.view(-1)[chosen_locations]
        errors = 1 - giou.gather(1, top)
        geometry = (errors * weights).sum() / weights.sum().clamp_min(1)
    bce = F.binary_cross_entropy_with_logits(logits, quality, reduction="none")
    classification = (bce * (logits.sigmoid() - quality).square()).sum() / quality.sum().clamp_min(1)
    return classification + 2 * geometry
