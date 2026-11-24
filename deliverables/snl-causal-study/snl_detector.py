"""Torchvision detector training and tiled inference for the SNL causal experiment."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models.detection import FCOS, FasterRCNN, RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.fcos import FCOSHead
from torchvision.ops import nms

from snl_inputs import DEFAULT_OUTPUT, WEIGHTS_FILENAME, sha_file

TILE = 192
OVERLAP = 48
CHECKPOINT_EPOCHS = (4, 8, 12)
FULL_FRAME_ARCHITECTURES = ("jiang-faster-rcnn", "full-small-anchors")


def backend():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def positions(size, tile=TILE, overlap=OVERLAP):
    if size < tile:
        raise ValueError("Radar crop is smaller than the detector tile")
    last = size - tile
    return sorted(set((*range(0, last + 1, tile - overlap), last)))


def tile_origins(height, width):
    return [(x, y) for y in positions(height) for x in positions(width)]


def tile_tensor(array, channels):
    if channels == "gray":
        array = np.repeat(array[:1], 3, axis=0)
    return torch.from_numpy(array.copy()).float().div_(255)


def create_detector(scale, *, official=None, architecture="faster-rcnn"):
    full_frame = architecture in FULL_FRAME_ARCHITECTURES
    size = TILE * scale
    anchors = ((12,), (20,), (30,), (48,), (64,)) if scale == 2 else ((12,), (18,), (27,), (42,), (72,))
    backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=None, trainable_layers=3 if full_frame else 2)
    options = {}
    if not full_frame:
        options = dict(min_size=size, max_size=size,
                       rpn_anchor_generator=AnchorGenerator(anchors, ((0.5, 1.0, 2.0),) * len(anchors)),
                       rpn_pre_nms_top_n_train=600, rpn_post_nms_top_n_train=250,
                       rpn_pre_nms_top_n_test=240, rpn_post_nms_top_n_test=100,
                       box_batch_size_per_image=128, box_detections_per_img=60)
    elif architecture == "full-small-anchors":
        options["rpn_anchor_generator"] = AnchorGenerator(((12,), (20,), (32,), (64,), (128,)),
                                                           ((0.67, 1.50, 2.50),) * 5)
    model = FasterRCNN(backbone, num_classes=91 if official else 2, box_score_thresh=0.005, **options)
    model.full_frame_inference = full_frame
    if official is not None:
        if not sha_file(Path(official)).startswith("258fb6c6"):
            raise ValueError("Official Torchvision checkpoint SHA-256 prefix differs")
        model.load_state_dict(torch.load(official, map_location="cpu", weights_only=True))
        model.roi_heads.box_predictor = FastRCNNPredictor(
            model.roi_heads.box_predictor.cls_score.in_features, 2)
    return model


def dense_backbone(official):
    backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=None, trainable_layers=2)
    if official is not None:
        if not sha_file(official).startswith("258fb6c6"):
            raise ValueError("Official Torchvision checkpoint SHA-256 prefix differs")
        ckpt = torch.load(official, map_location="cpu", weights_only=True)
        backbone.load_state_dict({k.removeprefix("backbone."): v for k, v in ckpt.items()
                                  if k.startswith("backbone.")})
    return backbone


def create_fcos(scale, *, official=None):
    size = TILE * scale
    backbone = dense_backbone(official)
    return FCOS(backbone, num_classes=2, min_size=size, max_size=size,
                anchor_generator=AnchorGenerator(((4,), (8,), (16,), (32,), (64,)), ((1.0,),)*5),
                head=FCOSHead(backbone.out_channels, 1, 2, num_convs=2),
                score_thresh=0.005, nms_thresh=0.6, detections_per_img=120, topk_candidates=350)


def create_retinanet(scale, *, official=None):
    size = TILE * scale
    # P2 is the finest FPN level. These sizes are in resized tile pixels.
    sizes = tuple(tuple(scale * base * 2**level for base in (7, 11, 16)) for level in range(5))
    anchors = AnchorGenerator(sizes, ((0.67, 1.0, 2.0),) * len(sizes))
    return RetinaNet(dense_backbone(official), num_classes=2, min_size=size, max_size=size,
                     anchor_generator=anchors, score_thresh=0.005, nms_thresh=0.60,
                     detections_per_img=70, topk_candidates=200)


DENSE_ARCHITECTURES = {"fcos": create_fcos, "retinanet": create_retinanet}


def target_in_tile(boxes, left, top):
    shift = np.array((left, top, left, top), dtype=np.float32)
    raw = boxes - shift
    clipped = np.clip(raw, 0, TILE)
    size = clipped[:, 2:] - clipped[:, :2]
    available = size.prod(axis=1)
    return clipped[(np.min(size, axis=1) >= 2) &
                   (available >= 0.5 * ((boxes[:, 2:] - boxes[:, :2]).prod(axis=1)))]


def draw_windows(boxes, width, height, rng, weights=None):
    if len(boxes):
        position = rng.integers(len(boxes)) if weights is None else rng.choice(len(boxes), p=weights)
        box = boxes[position]
        center = (box[:2] + box[2:]) / 2 - TILE / 2 + rng.integers(-45, 46, size=2)
        low = np.maximum(0, np.ceil(box[2:] - TILE)).astype(int)
        high = np.minimum(np.floor(box[:2]), [width - TILE, height - TILE]).astype(int)
        if np.any(low > high):
            raise ValueError("A training box does not fit into the tile")
        left, top = np.clip(np.rint(center).astype(int), low, high)
    else:
        left, top = rng.integers([0, 0], [width - TILE + 1, height - TILE + 1])
    return int(left), int(top)


def negative_window(boxes, width, height, rng):
    for _ in range(32):
        left, top = rng.integers([0, 0], [width - TILE + 1, height - TILE + 1])
        if not len(target_in_tile(boxes, left, top)):
            return int(left), int(top)
    return None


def training_tiles(ids, labels, width, height, rng, epoch, hard, *, object_weights=None):
    frames = list(ids[epoch % 2::2])
    rng.shuffle(frames)
    object_weights = object_weights or {}
    records = [(i, *draw_windows(labels[i], width, height, rng, object_weights.get(i))) for i in frames]
    for i in frames[::2]:
        region = negative_window(labels[i], width, height, rng)
        if region:
            records.append((i, *region))
    records.extend(hard[(epoch % 2)::2])
    rng.shuffle(records)
    return records


def detector_batch(records, features, labels, channels, rng, *, full_frame=False, tile_transform=None):
    images, targets = [], []
    for index, left, top in records:
        if full_frame:
            image = features[index - 2]
            boxes = labels[index].copy()
        else:
            image = features[index - 2, :, top:top + TILE, left:left + TILE]
            boxes = target_in_tile(labels[index], left, top)
        if tile_transform is not None:
            image = tile_transform(image, index, left, top)
        if rng.random() < 0.5:
            image = image[..., ::-1]
            boxes[:, [0, 2]] = image.shape[-1] - boxes[:, [2, 0]]
        images.append(tile_tensor(image, channels))
        targets.append({"boxes": torch.from_numpy(boxes), "labels": torch.ones(len(boxes), dtype=torch.int64)})
    return images, targets


def train_model(output, features, labels, kind, scale, *, hard=(), checkpoint_epochs=CHECKPOINT_EPOCHS,
                architecture="faster-rcnn", input_profile="raw"):
    seed = 1729
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = backend()
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    official = output / WEIGHTS_FILENAME
    if not official.is_file():
        official = DEFAULT_OUTPUT / WEIGHTS_FILENAME
    full_frame = architecture in FULL_FRAME_ARCHITECTURES
    if architecture in DENSE_ARCHITECTURES:
        model = DENSE_ARCHITECTURES[architecture](scale, official=official).to(device)
    else:
        model = create_detector(scale, official=official, architecture=architecture).to(device)
    # Keep the pretrained early image layers fixed and fine-tune upper ResNet stages,
    # FPN, RPN and the radar-specific detection head.
    params = [p for p in model.parameters() if p.requires_grad]
    if architecture in DENSE_ARCHITECTURES:
        optimizer = torch.optim.AdamW([
            {"params": (p for p in model.backbone.parameters() if p.requires_grad), "lr": 0.0002},
            {"params": model.head.parameters(), "lr": 0.001},
        ], weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(checkpoint_epochs), eta_min=0.00005)
    elif full_frame:
        optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        optimizer = torch.optim.SGD(params, lr=0.0025, momentum=0.9, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8], gamma=0.25)
    ids = tuple(sorted(labels))
    _, _, height, width = features.shape
    final_epoch = max(checkpoint_epochs)
    batch_size = 2 if full_frame else 4
    for epoch in range(1, final_epoch + 1):
        if full_frame:
            records = [(i, 0, 0) for i in rng.permutation(ids)]
        else:
            records = training_tiles(ids, labels, width, height, rng, epoch, hard)
        model.train()
        total_loss = 0.0
        updates = 0
        for start in range(0, len(records), batch_size):
            images, targets = detector_batch(records[start:start + batch_size], features, labels, kind, rng,
                                             full_frame=full_frame)
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            loss = sum(model(images, targets).values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"{kind}/{scale}x encountered a non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(params, 10.0)
            optimizer.step()
            total_loss += float(loss.detach())
            updates += 1
        scheduler.step()
        tag = kind
        if full_frame:
            tag = architecture
        elif architecture in DENSE_ARCHITECTURES:
            tag = f"{architecture}-{input_profile}-{kind}"
        print(f"{tag}/{scale}x epoch {epoch}/{final_epoch}: loss {total_loss / updates:.4f}", flush=True)
        if epoch in checkpoint_epochs:
            path = output / f"{tag}-{scale}x-epoch{epoch}.pt"
            torch.save(model.state_dict(), path)
            yield {"method": kind, "scale": scale, "epoch": epoch, "path": path.name, "sha256": sha_file(path),
                   "device": device.type, "torch": str(torch.__version__),
                   "architecture": architecture, "input_profile": input_profile}


def nonmax(boxes, scores, threshold):
    if len(boxes) == 0:
        return boxes, scores
    kept = nms(torch.from_numpy(boxes), torch.from_numpy(scores), threshold).numpy()
    return boxes[kept], scores[kept]


def map_tile_boxes(boxes, scores, left, top, width, height):
    ctr = (boxes[:, :2] + boxes[:, 2:]) / 2
    # Each overlapping seam has a tile whose accepted center range covers it.
    interior = ((ctr[:, 0] >= 12) | (left == 0)) & ((ctr[:, 1] >= 12) | (top == 0))
    interior &= ((ctr[:, 0] <= TILE - 12) | (left + TILE == width))
    interior &= ((ctr[:, 1] <= TILE - 12) | (top + TILE == height))
    native = boxes[interior] + np.array([left, top, left, top], dtype=np.float32)
    native = np.clip(native, 0, np.array([width, height, width, height], dtype=np.float32))
    valid = np.all(native[:, 2:] > native[:, :2], axis=1)
    return native[valid], scores[interior][valid]


def predict_frame(model, channels, kind):
    if hasattr(model, "predict_boxes"):
        if not getattr(model, "tile_inference", False):
            return model.predict_boxes(channels)
        height, width = channels.shape[-2:]
        partitions = []
        for left, top in tile_origins(height, width):
            tile = channels[:, top:top + TILE, left:left + TILE]
            boxes, scores = model.predict_boxes(tile)
            partitions.append(map_tile_boxes(boxes, scores, left, top, width, height))
        return np.concatenate([p[0] for p in partitions]), np.concatenate([p[1] for p in partitions])
    device = next(model.parameters()).device
    if getattr(model, "full_frame_inference", False):
        with torch.inference_mode():
            result = model([tile_tensor(channels, kind).to(device)])[0]
            keep = result["labels"] == 1
            return result["boxes"][keep].cpu().numpy(), result["scores"][keep].cpu().numpy()
    height, width = channels.shape[-2:]
    origins = tile_origins(height, width)
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(origins), 6):
            subset = origins[start:start + 6]
            inputs = [tile_tensor(channels[:, top:top+TILE, left:left+TILE], kind).to(device)
                      for left, top in subset]
            for result, (left, top) in zip(model(inputs), subset, strict=True):
                keep = result["labels"] == 1
                predictions.append(map_tile_boxes(result["boxes"][keep].cpu().numpy(), result["scores"][keep].cpu().numpy(), left, top,
                                                  width, height))
    return np.concatenate([p[0] for p in predictions]), np.concatenate([p[1] for p in predictions])


def load_trained(output, config):
    path = output / config["path"]
    if sha_file(path) != config["sha256"]:
        raise ValueError(f"{path.name}: checkpoint digest differs")
    if config.get("architecture") == "ultralytics-yolo11":
        from snl_yolo import YoloPredictor

        return YoloPredictor(output, path, config)
    if config.get("architecture") == "radar-center":
        from snl_center import CenterDetector

        model = CenterDetector(dense_offsets=bool(config.get("dense_offsets", False)),
                               input_channels=config.get("input_channels",3),
                               global_context=bool(config.get("global_context", False)),
                               learned_background=bool(config.get("learned_background", False)),
                               space_to_depth=bool(config.get("space_to_depth", False)),
                               shared_temporal=bool(config.get("shared_temporal", False)),
                               recurrent_temporal=bool(config.get("recurrent_temporal", False)),
                               mixed_temporal=bool(config.get("mixed_temporal", False)),
                               task_alignment=bool(config.get("task_alignment", False)),
                               line_context=bool(config.get("line_context", False)), backbone=config.get("backbone", "tiny"), pretrained_backbone=False,
                               channel_adapter=config.get("channel_adapter") if "channel_adapter" in config else None)
        model.tile_inference = bool(config.get("tiled", False))
    elif config.get("architecture") in DENSE_ARCHITECTURES:
        model = DENSE_ARCHITECTURES[config["architecture"]](config["scale"])
    elif config.get("architecture") == "radar-center-map-ensemble":
        from snl_center import CenterMapEnsemble

        model = CenterMapEnsemble(config["members"], config["input_channels"], backbone=config.get("backbone", "tiny"))
    else:
        model = create_detector(config["scale"], architecture=config.get("architecture", "faster-rcnn"))
    if config.get("flip_inference"):
        model.flip_inference = True
    if config.get("occupancy_power"):
        model.occupancy_power = config["occupancy_power"]
    if config.get("contrast_gain"):
        model.contrast_gain = config["contrast_gain"]
    if config.get("stitch_inference"):
        model.stitch_inference = True
    if config.get("dense_candidate_limit"):
        model.dense_candidate_limit = config["dense_candidate_limit"]
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(backend()).eval()
