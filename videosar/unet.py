from hashlib import sha256
from pathlib import Path

import numpy as np

from .center_box import CENTER_OUTPUTS, center_loss
from .proposal_data import translate


UNET = "smp.Unet/resnet18/depth3"
FPN = "smp.FPN/resnet18/depth3"
LRASPP = "torchvision.LRASPP/MobileNetV3"
CONTRAST_WINDOWS = ((3, 9), (5, 15))


def runtime():
    try:
        import segmentation_models_pytorch as smp
        import torch
        from skimage.registration import phase_cross_correlation
    except (ImportError, OSError) as exc:
        raise ValueError("U-Net requires the optional dependencies in requirements-unet.txt") from exc
    return torch, smp, phase_cross_correlation


def align(frames, correlate):
    registered = np.asarray(frames, dtype=np.float32).copy()
    offsets = np.zeros((len(frames), 2))
    window = (np.hanning(frames.shape[1])[:, None] * np.hanning(frames.shape[2])).astype(np.float32)
    log = np.log1p(frames)
    reference = (log[0] - log[0].mean()) * window
    if np.linalg.norm(reference) < 1e-6:
        return registered, offsets
    for t in range(1, len(frames)):
        moving = (log[t] - log[t].mean()) * window
        if np.linalg.norm(moving) < 1e-6:
            continue
        y, x = correlate(reference, moving, upsample_factor=5, normalization=None)[0]
        if max(abs(x), abs(y)) <= 2.2 and (x or y):
            offsets[t] = [x, y]
            registered[t] = translate(frames[t], x, y)
    return registered, offsets


def input_features(frames, temporal, history_frames=0, *, local_contrast=False):
    power = np.log1p(frames)
    if not temporal:
        return power[:, None]
    difference = np.zeros_like(power)
    difference[1:] = power[1:] - power[:-1]
    channels = [power, difference]
    if history_frames:
        residual = np.zeros_like(power)
        for i in range(1, len(power)):
            residual[i] = power[i] - np.median(power[max(0, i - history_frames):i], axis=0)
        channels.append(residual)
    if local_contrast:
        from scipy.ndimage import uniform_filter

        for inner, outer in CONTRAST_WINDOWS:
            center = uniform_filter(power, size=(1, inner, inner), mode="reflect")
            surrounding = uniform_filter(power, size=(1, outer, outer), mode="reflect")
            ring = (outer**2 * surrounding - inner**2 * center) / (outer**2 - inner**2)
            channels.append(ring - center)
    return np.stack(channels, axis=1)


def training_stats(arrays):
    counts = sum(len(a) * a.shape[-1] * a.shape[-2] for a in arrays)
    total = sum(a.sum(axis=(0, 2, 3), dtype=np.float64) for a in arrays)
    squares = sum(np.square(a, dtype=np.float64).sum(axis=(0, 2, 3)) for a in arrays)
    mean = total / counts
    std = np.sqrt(np.maximum(squares / counts - mean**2, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(features, mean, std):
    return (features - np.asarray(mean, dtype=np.float32)[None, :, None, None]) / np.asarray(std, dtype=np.float32)[None, :, None, None]


def create_model(torch, smp, temporal, architecture=UNET, history_frames=0, *, local_contrast=False, center_head=False):
    channels = 2 + int(bool(history_frames)) + len(CONTRAST_WINDOWS) * local_contrast if temporal else 1
    if architecture == LRASPP:
        from torchvision.models.segmentation import lraspp_mobilenet_v3_large

        model = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=1)
        stem = model.backbone["0"][0]
        model.backbone["0"][0] = torch.nn.Conv2d(channels, stem.out_channels, stem.kernel_size,
                                               stride=stem.stride, padding=stem.padding, bias=False)
        return model
    common = {"encoder_name": "resnet18", "encoder_depth": 3, "encoder_weights": None,
              "in_channels": channels, "classes": CENTER_OUTPUTS if center_head else 1}
    if architecture == UNET:
        model = smp.Unet(**common, decoder_channels=(48, 24, 12))
    elif architecture == FPN:
        model = smp.FPN(**common, decoder_pyramid_channels=64, decoder_segmentation_channels=32, upsampling=1)
    else:
        raise ValueError(f"unsupported segmentation architecture: {architecture}")
    # These deeper ResNet stages are unreachable at depth 3 and need no weights in the checkpoint.
    model.encoder.layer3 = torch.nn.Identity()
    model.encoder.layer4 = torch.nn.Identity()
    return model


def logits(torch, model, batch, input_scale=1, *, all_channels=False):
    height, width = batch.shape[-2:]
    scaled_h, scaled_w = height * input_scale, width * input_scale
    functional = torch.nn.functional
    if input_scale != 1:
        batch = functional.interpolate(batch, size=(scaled_h, scaled_w), mode="bilinear", align_corners=False)
    padded = functional.pad(batch, (0, -scaled_w % 8, 0, -scaled_h % 8), mode="replicate")
    output = model(padded)
    if isinstance(output, dict):
        output = output["out"]
    output = output[:, :, :scaled_h, :scaled_w]
    if input_scale != 1:
        output = functional.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)
    return output if all_channels else output[:, 0]


def loss_value(torch, output, labels, small_area_reference=None):
    p = output.sigmoid()
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits
    positive_weight = output.new_tensor(3.0)
    if small_area_reference is None:
        bce = bce_loss(output, labels, pos_weight=positive_weight)
        overlap = 2 * (p * labels).sum() + 1
        return bce + (1 - overlap / (p.sum() + labels.sum() + 1))
    area = labels.sum(dim=(1, 2), keepdim=True)
    frame_weight = (small_area_reference / area.clamp_min(1)).sqrt().clamp(1, 3)
    frame_weight = torch.where(area > 0, frame_weight, 1.0)
    bce = bce_loss(output, labels, pos_weight=positive_weight, reduction="none")
    bce = (bce * frame_weight).sum() / (frame_weight.sum() * labels.shape[1] * labels.shape[2])
    overlap = 2 * (p * labels * frame_weight).sum() + 1
    return bce + (1 - overlap / (((p + labels) * frame_weight).sum() + 1))


def noisy_features(frames, rng, temporal, history_frames=0, *, local_contrast=False):
    sigma = rng.uniform(0.045, 0.20)
    amplitude = np.sqrt(frames)
    noise = rng.normal(0, sigma / np.sqrt(2), size=(2, *frames.shape)).astype(np.float32)
    noisy = (amplitude + noise[0])**2 + noise[1]**2
    noisy *= np.exp(rng.normal(0, 0.12, frames.shape).astype(np.float32))
    return input_features(noisy, temporal, history_frames, local_contrast=local_contrast)


def fit(torch, smp, train, validation, *, temporal, augment_noise, epochs, seed, device, batch_size,
        initialization=None, learning_rate=0.001, architecture=UNET, input_scale=1,
        history_frames=0, weight_small_shadows=False, local_contrast=False, center_head=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    features = [input_features(frames, temporal, history_frames, local_contrast=local_contrast) for frames, _ in train]
    if initialization is None:
        mean, std = training_stats(features)
        model = create_model(torch, smp, temporal, architecture, history_frames,
                             local_contrast=local_contrast, center_head=center_head)
        model = model.to(device, memory_format=torch.channels_last)
    else:
        model, previous = initialization
        mean, std = np.asarray(previous["mean"], dtype=np.float32), np.asarray(previous["std"], dtype=np.float32)
        if len(mean) < features[0].shape[1]:
            added_mean, added_std = training_stats(features)
            mean, std = np.append(mean, added_mean[len(mean):]), np.append(std, added_std[len(std):])
    train_x = torch.from_numpy(normalize(np.concatenate(features), mean, std))
    train_y = torch.from_numpy(np.concatenate([masks for _, masks in train]).astype(np.float32))
    area_reference = None
    if weight_small_shadows:
        areas = train_y.sum(dim=(1, 2))
        area_reference = float(torch.quantile(areas[areas > 0], 0.75))
    val_x = torch.from_numpy(normalize(np.concatenate([
        input_features(frames, temporal, history_frames, local_contrast=local_contrast)
        for frames, _ in validation]), mean, std))
    val_y = torch.from_numpy(np.concatenate([masks for _, masks in validation]).astype(np.float32))
    if augment_noise:
        noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xA38]))
        noisy = [noisy_features(frames, noise_rng, temporal, history_frames, local_contrast=local_contrast)
                 for frames, _ in train]
        noisy_x = torch.from_numpy(normalize(np.concatenate(noisy), mean, std))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)
    history, best_state = [], None
    best, best_epoch = float("inf"), 0

    def batch_loss(x, y):
        output = logits(torch, model, x, input_scale, all_channels=center_head)
        if center_head:
            return loss_value(torch, output[:, 0], y, area_reference) + center_loss(torch, output, y)
        return loss_value(torch, output, y, area_reference)

    for epoch in range(epochs):
        model.train()
        training_loss = 0.0
        order = rng.permutation(len(train_x))
        for i in range(0, len(order), batch_size):
            index = order[i:i + batch_size]
            x, y = train_x[index].to(device), train_y[index].to(device)
            if augment_noise:
                use_noise = torch.from_numpy(noise_rng.random(len(index)) < 0.5).to(device)
                x = torch.where(use_noise[:, None, None, None], noisy_x[index].to(device), x)
            if rng.random() < 0.5:
                x, y = x.flip((-1,)), y.flip((-1,))
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(x, y)
            loss.backward()
            optimizer.step()
            training_loss += float(loss.detach()) * len(index)
        model.eval()
        validation_loss = 0.0
        with torch.inference_mode():
            for x, y in zip(val_x.split(batch_size), val_y.split(batch_size)):
                validation_loss += float(batch_loss(x.to(device), y.to(device))) * len(x)
        validation_loss /= len(val_x)
        schedule.step()
        if not np.isfinite(validation_loss):
            raise ValueError("U-Net validation loss is non-finite")
        history.append({"epoch": epoch + 1, "train_loss": training_loss / len(train_x), "validation_loss": validation_loss})
        print(f"epoch {epoch + 1}/{epochs}  train {training_loss / len(train_x):.4f}  val {validation_loss:.4f}", flush=True)
        if validation_loss < best - 1e-4:
            best, best_epoch = validation_loss, epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch + 1 - best_epoch >= 5:
            break
    model.load_state_dict(best_state)
    return model, {"format_version": 1, "architecture": architecture, "input_scale": input_scale,
                   "history_frames": history_frames, "local_contrast": local_contrast, "center_head": center_head,
                   "small_area_reference": area_reference,
                   "temporal": temporal, "noise_augmentation": augment_noise, "mean": mean.tolist(), "std": std.tolist(),
                   "best_epoch": best_epoch, "history": history,
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def predict(torch, model, aligned, settings, device, batch_size, *, all_channels=False):
    features = input_features(aligned, settings["temporal"], settings.get("history_frames", 0),
                              local_contrast=settings.get("local_contrast", False))
    x = normalize(features, settings["mean"], settings["std"])
    model.eval()
    values = []
    with torch.inference_mode():
        for batch in torch.from_numpy(x).split(batch_size):
            output = logits(torch, model, batch.to(device), settings.get("input_scale", 1), all_channels=all_channels)
            values.append((output if all_channels else output.sigmoid()).cpu().numpy())
    return np.concatenate(values)


def predict_centers(torch, model, aligned, settings, device, batch_size):
    from scipy.special import expit

    output = predict(torch, model, aligned, settings, device, batch_size, all_channels=True)
    return expit(output[:, 1]), output[:, 2:]


def load_weights(torch, smp, path, settings, device):
    architecture = settings.get("architecture")
    history_frames = settings.get("history_frames", 0)
    temporal = settings.get("temporal")
    local_contrast = settings.get("local_contrast", False)
    center_head = settings.get("center_head", False)
    detection_mode = settings.get("detection_mode", "segmentation")
    if (settings.get("format_version") != 1 or architecture not in (UNET, FPN, LRASPP)
            or settings.get("input_scale", 1) not in (1, 2) or history_frames not in (0, 5)
            or type(local_contrast) is not bool
            or type(center_head) is not bool or detection_mode not in ("segmentation", "center_box")
            or (detection_mode == "center_box" and not center_head)
            or ((history_frames or local_contrast or center_head) and (not temporal or architecture != UNET))):
        raise ValueError("unsupported segmentation checkpoint settings")
    channels = 2 + int(bool(history_frames)) + len(CONTRAST_WINDOWS) * local_contrast if temporal else 1
    mean, std = np.asarray(settings["mean"]), np.asarray(settings["std"])
    if type(temporal) is not bool or mean.shape != (channels,) or std.shape != (channels,) or not (
            np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()):
        raise ValueError("invalid U-Net normalization or temporal settings")
    path = Path(path)
    if sha256(path.read_bytes()).hexdigest() != settings.get("weight_sha256"):
        raise ValueError("U-Net checkpoint checksum differs from its model settings")
    model = create_model(torch, smp, temporal, architecture, history_frames,
                         local_contrast=local_contrast, center_head=center_head)
    model = model.to(device, memory_format=torch.channels_last)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model
