import math


SIMULATION = {
    "scene": "geometry", "background": "clutter", "frames": 24, "width": 96, "height": 64,
    "frame_interval_s": 0.2, "pixel_size_m": 0.25, "noise": 0.12, "thermal_noise": 0.0, "seed": 42,
    "shadow_width_px": 32, "shadow_height_px": 16, "shadow_factor": 0.12,
    "object_width_m": 0.9, "object_depth_m": 1.3, "object_height_m": 1.6,
    "start_x_m": -6.0, "start_y_m": -3.0, "velocity_x_mps": 2.0, "velocity_y_mps": 0.35,
    "radar_height_m": 30.0, "radar_offset_m": 40.0, "platform_speed_mps": 2.4,
    "carrier_hz": 35e9, "bandwidth_hz": 600e6, "pulses_per_frame": 41,
    "pulse_interval_s": 0.005, "visibility_drop": 0.25,
}
DETECTION = {"method": "threshold", "threshold": None, "min_area": 4}
TRACKING = {"gate_px": 12.0, "max_gap": 2}
EVALUATION = {"match_iou": 0.3}

CHOICES = {"scene": ("legacy", "geometry", "empty"), "background": ("uniform", "texture", "clutter", "roads"),
           "method": ("threshold", "difference", "learned"), "profile": ("mixed", "sar", "proposal"),
           "device": ("auto", "cpu", "cuda"),
           "network": ("compact", "context")}


def merged(defaults, supplied):
    unknown = set(supplied) - defaults.keys()
    if unknown:
        raise ValueError(f"unknown configuration fields: {', '.join(sorted(unknown))}")
    values = defaults | supplied
    if values.get("profile") == "proposal" and "dataset_size" not in supplied:
        values["dataset_size"] = 120
    if values.get("scene") == "legacy":
        legacy = {"width": 192, "height": 128, "noise": 0.0, "shadow_factor": 0.25, "background": "uniform"}
        values.update({key: value for key, value in legacy.items() if key not in supplied})
    for key, default in defaults.items():
        value = values[key]
        if key in CHOICES:
            if value not in CHOICES[key]:
                raise ValueError(f"{key} must be one of {', '.join(CHOICES[key])}")
        elif default is None:
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError(f"{key} must be a finite number or null")
        elif type(default) is int:
            if type(value) is not int:
                raise ValueError(f"{key} must be an integer")
        elif type(default) is float and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError(f"{key} must be a finite number")
    validate(values)
    return values


def validate(config):
    ranges = {"frames": (1, 512), "width": (8, 1024), "height": (8, 1024),
              "frame_interval_s": (0.001, 60), "pixel_size_m": (0.01, 20), "noise": (0, 2),
              "thermal_noise": (0, 2), "seed": (0, 2**32 - 1),
              "shadow_width_px": (1, 1024), "shadow_height_px": (1, 1024), "shadow_factor": (0, 1),
              "object_width_m": (0.01, 100), "object_depth_m": (0.01, 100), "object_height_m": (0.01, 100),
              "radar_height_m": (0.1, 10_000), "radar_offset_m": (0.1, 10_000), "platform_speed_mps": (0.01, 200),
              "carrier_hz": (26.5e9, 40e9), "bandwidth_hz": (1e6, 2e9), "pulses_per_frame": (3, 401),
              "pulse_interval_s": (0.00001, 1), "visibility_drop": (0, 1), "threshold": (0.000001, 1),
              "min_area": (1, 1_048_576), "gate_px": (0.01, 2048), "max_gap": (0, 100),
              "match_iou": (0.000001, 1), "epochs": (1, 300), "batch_size": (2, 128), "dataset_size": (12, 300)}
    for field, (lower, upper) in ranges.items():
        value = config.get(field)
        if value is not None and not lower <= value <= upper:
            raise ValueError(f"{field} must be between {lower:g} and {upper:g}")
    if "pulses_per_frame" not in config:
        return
    if config["pulses_per_frame"] % 2 != 1:
        raise ValueError("pulses_per_frame must be odd")
    if config["object_height_m"] >= config["radar_height_m"]:
        raise ValueError("object_height_m must be smaller than radar_height_m")
    if config["width"] * config["height"] * config["frames"] > 32_000_000:
        raise ValueError("width * height * frames exceeds 32 million pixels")
    if config["scene"] == "legacy":
        if config.get("thermal_noise", 0):
            raise ValueError("thermal_noise is supported by the geometric and SAR scene models")
        last = config["frames"] - 1
        if (8 + 6 * last + config["shadow_width_px"] > config["width"] or
                48 + last + config["shadow_height_px"] > config["height"]):
            raise ValueError("legacy rectangle falls outside the frame; adjust frames or dimensions")
