import numpy as np


C = 299_792_458.0


def box_mean(image, radius):
    if not radius:
        return image
    padded = np.pad(image, ((radius, radius), (radius, radius)), mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0, dtype=np.float64).cumsum(1)
    size = 2 * radius + 1
    return ((integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size]
             + integral[:-size, :-size]) / size**2).astype(np.float32)


def grid(config):
    scale = config["pixel_size_m"]
    x = (np.arange(config["width"]) - (config["width"] - 1) / 2) * scale
    y = (np.arange(config["height"]) - (config["height"] - 1) / 2) * scale
    return np.meshgrid(x, y)


def occluded(x, y, radar_x, center, config):
    if config["scene"] == "empty":
        return np.zeros(x.shape, dtype=bool)
    # Intersect the ground-to-radar segment with the prism below its top face.
    low = np.zeros_like(x)
    high = np.full_like(x, config["object_height_m"] / config["radar_height_m"])
    for ground, radar, origin, width in ((x, radar_x, center[0], config["object_width_m"]),
                                         (y, -config["radar_offset_m"], center[1], config["object_depth_m"])):
        direction = radar - ground
        coincident = np.abs(direction) < 1e-12
        safe_direction = np.where(coincident, 1.0, direction)
        start = (origin - width / 2 - ground) / safe_direction
        end = (origin + width / 2 - ground) / safe_direction
        inside = np.abs(ground - origin) <= width / 2
        near = np.where(coincident, np.where(inside, -np.inf, np.inf), np.minimum(start, end))
        far = np.where(coincident, np.where(inside, np.inf, -np.inf), np.maximum(start, end))
        low = np.maximum(low, near)
        high = np.minimum(high, far)
    return low <= high


def positions(config, time):
    return (config["start_x_m"] + config["velocity_x_mps"] * time,
            config["start_y_m"] + config["velocity_y_mps"] * time)


def background(x, y, config, rng):
    if config["background"] == "uniform":
        return np.ones_like(x)
    phase = rng.uniform(0, 2 * np.pi, 3)
    terrain = 1 + 0.18 * np.sin(0.55 * x + phase[0]) + 0.15 * np.cos(0.73 * y + phase[1])
    terrain += 0.13 * np.sin(0.35 * x + 0.44 * y + phase[2])
    if config["background"] == "clutter":
        cx, cy = rng.uniform(-0.25, 0.25, 2) * [np.ptp(x), np.ptp(y)]
        terrain[(abs(x - cx) < 1.4) & (abs(y - cy) < 1.1)] *= 0.18
    if config["background"] == "roads":
        lane = y - rng.uniform(-0.45, 0.45) * x - rng.uniform(-2, 2)
        terrain[abs(lane) < 0.65] *= 0.20
        terrain[(abs(lane) >= 0.65) & (abs(lane) < 0.95)] *= 1.35
        terrain[(abs(x - rng.uniform(-4, 4)) < 1.1) & (abs(y - rng.uniform(-3, 3)) < 1.5)] *= 0.30
    return terrain


def metadata(config, times, source):
    result = {"source": source, "timestamps_s": times.tolist(), "frame_interval_s": config["frame_interval_s"],
              "pixel_size_m": config["pixel_size_m"], "intensity_unit": "relative_linear_power",
              "coordinate_convention": "frames[t,y,x]; x along track; y away from the radar side",
              "seed": config["seed"], "effective_config": config}
    if config["scene"] != "empty":
        result["target_centers_m"] = np.asarray(positions(config, times)).T.tolist()
        result["target_velocity_mps"] = [config["velocity_x_mps"], config["velocity_y_mps"]]
    return result


def simulate(config, ground_speckle=False):
    times = np.arange(config["frames"]) * config["frame_interval_s"]
    if config["scene"] == "legacy":
        frames = np.full((len(times), config["height"], config["width"]), 180, dtype=np.float32)
        masks = np.zeros(frames.shape, dtype=bool)
        for t, mask in enumerate(masks):
            x, y = 8 + 6 * t, 48 + t
            mask[y:y + config["shadow_height_px"], x:x + config["shadow_width_px"]] = True
        frames[masks] *= config["shadow_factor"]
        rng = np.random.default_rng(config["seed"])
        if config["noise"]:
            frames = np.maximum(0, frames + rng.normal(0, 180 * config["noise"], frames.shape))
        return frames, masks, {"source": "legacy_rectangular_attenuation", "timestamps_s": times.tolist(),
                               "frame_interval_s": config["frame_interval_s"], "pixel_size_m": None,
                               "intensity_unit": "arbitrary_linear_unit", "seed": config["seed"],
                               "coordinate_convention": "frames[t,y,x]; x right; y down",
                               "effective_config": config}, None

    x, y = grid(config)
    rng = np.random.default_rng(config["seed"])
    ground = background(x, y, config, rng)
    if ground_speckle:
        ground *= rng.gamma(shape=3, scale=1 / 3, size=x.shape)
    persistent = rng.normal(0, config["noise"] / 2, x.shape)
    frames, masks = [], []
    for time in times:
        mask = occluded(x, y, config["platform_speed_mps"] * time, positions(config, time), config)
        frame = ground * np.where(mask, config["shadow_factor"], 1)
        frame *= np.exp(persistent + rng.normal(0, config["noise"], x.shape))
        if config.get("thermal_noise", 0):
            noise = (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)) * (config["thermal_noise"] / np.sqrt(2))
            frame = abs(np.sqrt(frame) + noise)**2
        frames.append(frame)
        masks.append(mask)
    return np.asarray(frames), np.asarray(masks), metadata(config, times, "geometric_prism"), None


class RangeProcessor:
    def __init__(self, x, y, pulse_x, config):
        self.config = config
        self.spacing = C / (4 * config["bandwidth_hz"])
        corners_x = [float(np.min(x)), float(np.max(x))]
        corners_y = [float(np.min(y)), float(np.max(y))]
        distances = [np.hypot(np.hypot(cx - px, cy + config["radar_offset_m"]), config["radar_height_m"])
                     for cx in corners_x for cy in corners_y for px in (min(pulse_x), max(pulse_x))]
        nearest_x = max(corners_x[0] - max(pulse_x), min(pulse_x) - corners_x[1], 0)
        nearest_y = max(corners_y[0] + config["radar_offset_m"], -config["radar_offset_m"] - corners_y[1], 0)
        self.origin = np.hypot(np.hypot(nearest_x, nearest_y), config["radar_height_m"]) - 4 * self.spacing
        self.bins = int(np.ceil((max(distances) - self.origin) / self.spacing)) + 5
        # At Fs=2B, a 32-sample baseband LFM pulse sweeps across B.
        time = (np.arange(32) - 15.5) / (2 * config["bandwidth_hz"])
        rate = config["bandwidth_hz"] / (32 / (2 * config["bandwidth_hz"]))
        self.chirp = np.exp(1j * np.pi * rate * time**2)

    def ranges(self, x, y, pulse_x):
        return np.sqrt((x - pulse_x)**2 + (y + self.config["radar_offset_m"])**2 + self.config["radar_height_m"]**2)

    def coordinates(self, distances):
        bins = (distances - self.origin) / self.spacing
        phase = np.exp(-4j * np.pi * self.config["carrier_hz"] * distances / C)
        return bins, phase

    def compress(self, bins, amplitudes):
        lower = np.floor(bins).astype(np.intp)
        fraction = bins - lower
        impulses = np.zeros(self.bins, dtype=complex)
        np.add.at(impulses, lower.ravel(), (amplitudes * (1 - fraction)).ravel())
        np.add.at(impulses, (lower + 1).ravel(), (amplitudes * fraction).ravel())
        echo = np.convolve(impulses, self.chirp)
        return self.matched_filter(echo)

    def matched_filter(self, echo):
        return np.correlate(echo, self.chirp, mode="valid") / len(self.chirp)

    def backproject(self, profile, bins, phase):
        lower = np.floor(bins).astype(np.intp)
        fraction = bins - lower
        samples = (1 - fraction) * profile[lower] + fraction * profile[lower + 1]
        return samples * phase.conj()


def pulse_positions(config, times):
    offsets = (np.arange(config["pulses_per_frame"]) - config["pulses_per_frame"] // 2) * config["pulse_interval_s"]
    pulse_times = times[:, None] + offsets
    return pulse_times, pulse_times * config["platform_speed_mps"]


def _focus_fields(config, rng, static_dark, receiver_rng, thermal, *, processor=None, noise_echoes=None,
                  hold_target_within_frame=False, include_shadow_terms=False):
    x, y = grid(config)
    times = np.arange(config["frames"]) * config["frame_interval_s"]
    pulse_times, pulse_x = pulse_positions(config, times)
    if processor is None:
        processor = RangeProcessor(x, y, pulse_x.ravel(), config)
    ground = background(x, y, config, rng)
    if static_dark is not None:
        cx, cy, width, depth, factor = static_dark
        ground[(abs(x - cx) <= width / 2) & (abs(y - cy) <= depth / 2)] *= factor
    field = (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)) * np.sqrt(ground / 2)
    shadow_amplitude = np.sqrt(config["shadow_factor"])
    for frame, (time, aperture_times, aperture_x) in enumerate(zip(times, pulse_times, pulse_x)):
        reference = np.zeros(x.shape, dtype=complex)
        removed = np.zeros_like(reference)
        receiver_noise = np.zeros_like(reference) if thermal else 0
        hits = np.zeros(x.shape, dtype=float)
        for pulse, (pulse_time, px) in enumerate(zip(aperture_times, aperture_x)):
            target_time = time if hold_target_within_frame else pulse_time
            blocked = occluded(x, y, px, positions(config, target_time), config)
            bins, phase = processor.coordinates(processor.ranges(x, y, px))
            scatterers = field * phase
            reference += processor.backproject(processor.compress(bins, scatterers), bins, phase)
            if blocked.any() and shadow_amplitude < 1:
                blocked_profile = processor.compress(bins[blocked], scatterers[blocked])
                removed += (1 - shadow_amplitude) * processor.backproject(blocked_profile, bins, phase)
            if thermal:
                if noise_echoes is None:
                    samples = processor.bins + len(processor.chirp) - 1
                    noise = (receiver_rng.normal(size=samples) + 1j * receiver_rng.normal(size=samples)) * (thermal / np.sqrt(2))
                else:
                    noise = noise_echoes[frame, pulse]
                receiver_noise += processor.backproject(processor.matched_filter(noise), bins, phase)
            hits += blocked
        clean = reference - removed
        power = np.abs(clean / len(aperture_x))**2
        baseline = np.abs(reference / len(aperture_x))**2
        # Visibility is an operational label based on local counterfactual image contrast.
        locally_darkened = box_mean(power, 1) < (1 - config["visibility_drop"]) * box_mean(baseline, 1)
        blocked_support = hits >= len(aperture_x) / 2
        mask = blocked_support & locally_darkened
        geometry = occluded(x, y, config["platform_speed_mps"] * time, positions(config, time), config)
        output = (clean, receiver_noise, power, baseline, mask, geometry)
        yield output + (reference, removed, blocked_support) if include_shadow_terms else output


def form_video(config, normalize_power=True, static_dark=None):
    rng = np.random.default_rng(config["seed"])
    thermal = config.get("thermal_noise", 0)
    frames, masks, geometry, counterfactual = [], [], [], []
    count = config["pulses_per_frame"]
    for clean, receiver, power, baseline, mask, ground_mask in _focus_fields(config, rng, static_dark, rng, thermal):
        frames.append(np.abs((clean + receiver) / count)**2 if thermal else power)
        counterfactual.append(baseline)
        masks.append(mask)
        geometry.append(ground_mask)
    scale = max(float(np.mean(frames)), 1e-12) if normalize_power else 1.0
    frames = np.asarray(frames) / scale
    frames *= np.exp(rng.normal(0, config["noise"], frames.shape))
    times = np.arange(config["frames"]) * config["frame_interval_s"]
    info = metadata(config, times, "single_channel_sar_simulation")
    info.update({"sample_rate_hz": 2 * config["bandwidth_hz"], "slant_range_bin_m": C / (4 * config["bandwidth_hz"]),
                 "nominal_slant_resolution_m": C / (2 * config["bandwidth_hz"]),
                 "visibility_definition": "blocked for >=50% of pulses and local noise-free power drops by configured fraction",
                 "noise_model": "image-domain multiplicative log-normal perturbation" +
                                (" and complex Gaussian receiver noise" if config.get("thermal_noise", 0) else ""),
                 "power_normalization": "mean formed observation power before image-domain perturbation" if normalize_power else "none"})
    return frames, np.asarray(masks), info, np.asarray(geometry), np.asarray(counterfactual) / scale


def calibrated_receiver_video(config, snr_targets_db, static_dark=None):
    """Scale one focused complex receiver-noise realization across all requested SNRs."""
    ground_rng = np.random.default_rng(config["seed"])
    noise_rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 1021]))
    fields = _focus_fields(config, ground_rng, static_dark, noise_rng, thermal=1)
    return _receiver_levels(config, fields, snr_targets_db)


def _shared_aperture_noise(config, longest):
    stride = config["frame_interval_s"] / config["pulse_interval_s"]
    if not np.isclose(stride, round(stride), rtol=0, atol=1e-10):
        raise ValueError("frame centers must lie on the shared pulse train")
    frame_starts = np.arange(config["frames"]) * round(stride)
    times = np.arange(config["frames"]) * config["frame_interval_s"]
    _, pulse_x = pulse_positions(config | {"pulses_per_frame": longest}, times)
    processor = RangeProcessor(*grid(config), pulse_x.ravel(), config)
    noise_rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 1031]))
    size = (int(frame_starts[-1]) + longest, processor.bins + len(processor.chirp) - 1)
    train = (noise_rng.standard_normal(size) + 1j * noise_rng.standard_normal(size)) / np.sqrt(2)
    return processor, train, frame_starts


def paired_aperture_video(config, pulse_counts, snr_db, static_dark=None, *, hold_target_within_frame=False):
    """Yield centered subapertures of one pulse train at fixed formed-image SNR."""
    if not pulse_counts or any(count < 1 or count % 2 == 0 for count in pulse_counts):
        raise ValueError("paired apertures require positive odd pulse counts")
    longest = max(pulse_counts)
    processor, train, frame_starts = _shared_aperture_noise(config, longest)
    for count in pulse_counts:
        start = (longest - count) // 2
        pulse_indices = frame_starts[:, None] + start + np.arange(count)
        effective = config | {"pulses_per_frame": count}
        rng = np.random.default_rng(config["seed"])
        fields = _focus_fields(effective, rng, static_dark, None, thermal=1,
                               processor=processor, noise_echoes=train[pulse_indices],
                               hold_target_within_frame=hold_target_within_frame)
        yield effective, _receiver_levels(effective, fields, (snr_db,))[snr_db]


def paired_shadow_video(config, retentions, snr_db, static_dark=None):
    """Reuse one focused aperture and receiver noise across shadow attenuation levels."""
    if not retentions or any(not 0 < value <= 1 for value in retentions):
        raise ValueError("shadow attenuation retention must be in (0, 1]")
    count = config["pulses_per_frame"]
    processor, train, starts = _shared_aperture_noise(config, count)
    pulse_indices = starts[:, None] + np.arange(count)
    rng = np.random.default_rng(config["seed"])
    fields = list(_focus_fields(config, rng, static_dark, None, thermal=1,
                                processor=processor, noise_echoes=train[pulse_indices],
                                include_shadow_terms=True))
    original = config["shadow_factor"]
    original_amplitude_drop = 1 - np.sqrt(original)
    for retention in retentions:
        if retention == 1 or original_amplitude_drop == 0:
            yield retention, _receiver_levels(config, (frame[:6] for frame in fields), (snr_db,))[snr_db]
            continue
        shadow_factor = 1 - retention * (1 - original)
        amplitude_ratio = (1 - np.sqrt(shadow_factor)) / original_amplitude_drop
        weakened = []
        for _, noise, _, baseline, _, geometry, reference, removed, support in fields:
            clean = reference - amplitude_ratio * removed
            power = np.abs(clean / count)**2
            darkened = box_mean(power, 1) < (1 - config["visibility_drop"]) * box_mean(baseline, 1)
            weakened.append((clean, noise, power, baseline, support & darkened, geometry))
        effective = config | {"shadow_factor": shadow_factor}
        yield retention, _receiver_levels(effective, weakened, (snr_db,))[snr_db]


def _receiver_levels(config, fields, snr_targets_db):
    clean, noise, _, reference, masks, geometry = (np.asarray(values) for values in zip(*fields))
    clean /= config["pulses_per_frame"]
    noise /= config["pulses_per_frame"]
    reference_power, unit_noise_power = reference.mean(), np.mean(np.abs(noise)**2)
    times = np.arange(config["frames"]) * config["frame_interval_s"]
    outputs = {}
    for snr_db in snr_targets_db:
        scale = np.sqrt(reference_power / (unit_noise_power * 10**(snr_db / 10)))
        scaled_noise = scale * noise
        noise_power = np.mean(np.abs(scaled_noise)**2)
        info = metadata(config, times, "single_channel_sar_simulation")
        info.update(snr_target_db=snr_db, snr_measured_db=float(10 * np.log10(reference_power / noise_power)),
                    receiver_noise_scale=float(scale), receiver_noise_power=float(noise_power),
                    reference_mean_power=float(reference_power), power_normalization="none")
        outputs[snr_db] = (np.abs(clean + scaled_noise)**2, masks, info, geometry, reference)
    return outputs


def point_check(config, *, azimuth_offsets_m=None):
    x, y = grid(config)
    _, pulse_x = pulse_positions(config, np.array([0.0]))
    aperture = pulse_x[0]
    processor = RangeProcessor(x, y, aperture, config)
    iy, ix = x.shape[0] // 2 + 2, x.shape[1] // 2 - 3
    image = np.zeros(x.shape, dtype=complex)
    azimuth = None
    if azimuth_offsets_m is not None:
        cut_x = x[iy, ix] + np.asarray(azimuth_offsets_m)
        azimuth = np.zeros(cut_x.shape, dtype=complex)
    for index, px in enumerate(aperture):
        bins, phase = processor.coordinates(processor.ranges(x, y, px))
        profile = processor.compress(np.array([bins[iy, ix]]), np.array([phase[iy, ix]]))
        image += processor.backproject(profile, bins, phase)
        if azimuth is not None:
            cut_bins, cut_phase = processor.coordinates(processor.ranges(cut_x, y[iy, ix], px))
            azimuth += processor.backproject(profile, cut_bins, cut_phase)
        if index == len(aperture) // 2:
            range_error = abs(float(np.argmax(abs(profile)) - bins[iy, ix]))
            distance_m = processor.origin + float(bins[iy, ix]) * processor.spacing
    peak = np.unravel_index(np.argmax(abs(image)), image.shape)
    lag = np.arange(-31, 32)
    correlation = abs(np.correlate(processor.chirp, processor.chirp, mode="full")) / len(processor.chirp)
    dense_lag = np.linspace(-3, 3, 601)
    half_power = np.interp(dense_lag, lag, correlation) >= 1 / np.sqrt(2)
    width_m = float(np.ptp(dense_lag[half_power]) * processor.spacing)
    check = {"range_peak_error_bins": range_error, "range_peak_error_m": range_error * processor.spacing,
            "expected_echo_delay_s": 2 * distance_m / C, "delay_error_s": 2 * range_error * processor.spacing / C,
            "point_error_px": float(np.hypot(peak[0] - iy, peak[1] - ix)),
            "point_expected_yx": [iy, ix], "point_peak_yx": [int(v) for v in peak],
            "range_mainlobe_half_power_width_m": width_m}
    if azimuth is not None:
        check["azimuth_power"] = (np.abs(azimuth / len(aperture))**2).tolist()
    return check, np.abs(image / len(aperture))**2
