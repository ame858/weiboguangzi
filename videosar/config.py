"""Unified configuration schema for the refactored Video-SAR repository.

Key upgrades over legacy ``videosar/config.py``:
- Default spatial resolution raised from 96x64 to 256x256 (configurable up to 512x512+).
- Multi-target (3-15) curved/accelerating road trajectories & static building distractors.
- Explicit Microwave Photonics (MWP) degradation toggle so Ka-band Video-SAR and MWP
  ablation studies remain cleanly separated.
- Strict sequence-level splitting and object/tracking evaluation parameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple
import json
import yaml


@dataclass
class RadarConfig:
    """Ka-band (or configurable band) Video-SAR sensor geometry and waveform config."""

    carrier_freq_hz: float = 35.0e9          # Ka-band 35 GHz (matches SNL Video-SAR)
    bandwidth_hz: float = 600.0e6            # 600 MHz -> ~0.25 m slant-range resolution
    prf_hz: float = 1000.0                   # Pulse repetition frequency
    frame_rate_hz: float = 5.0               # Video-SAR frame rate (Hz)
    grazing_angle_deg: float = 30.0          # Depression / grazing angle
    slant_range_m: float = 4000.0            # Nominal slant range to scene center
    platform_speed_mps: float = 60.0         # Airborne platform speed
    pixel_spacing_m: float = 0.25            # Ground pixel spacing (meters / pixel)
    azimuth_look_dir_deg: float = 90.0       # Radar illumination direction in image plane
    n_looks: float = 4.0                     # Effective number of looks (speckle ENL)


@dataclass
class MWPConfig:
    """Optional Microwave Photonics (MWP) link degradation parameters.

    Disabled by default for pure Ka-band Video-SAR experiments. Enable explicitly
    when running microwave-photonics transceiver ablation studies.
    """

    enabled: bool = False
    laser_linewidth_hz: float = 10.0e3       # Lorentzian laser linewidth (Hz)
    mzm_modulation_index: float = 0.35       # Electro-optic MZM modulation depth (rad)
    mzm_bias_drift_rad: float = 0.02         # Quadrature bias point drift (rad)
    pd_responsivity_a_per_w: float = 0.8     # Photodetector responsivity (A/W)
    optical_power_mw: float = 10.0           # Optical carrier power at PD (mW)
    thermal_noise_std: float = 1.5e-3        # Receiver thermal noise floor (relative)
    include_shot_noise: bool = True
    include_phase_noise: bool = True
    include_mzm_nonlinearity: bool = True


@dataclass
class SimulationConfig:
    """High-fidelity Video-SAR simulation & real-background shadow injection config."""

    backend: Literal["real_bg_injection", "radarsimpy"] = "real_bg_injection"
    height: int = 256                        # Minimum 256 (replaces legacy 96x64)
    width: int = 256                         # Minimum 256 (replaces legacy 96x64)
    num_frames: int = 32
    min_targets: int = 3
    max_targets: int = 12
    num_static_distractors: int = 6          # Static building/tree occlusion shadows
    min_speed_mps: float = 4.0
    max_speed_mps: float = 18.0
    max_accel_mps2: float = 2.5
    road_curvature: float = 0.008            # Non-linear curved road trajectory factor
    shadow_contrast_range: Tuple[float, float] = (0.18, 0.45)
    doppler_smear_gain: float = 1.25         # Azimuth shift & defocusing smear strength
    enable_stop_and_go: bool = False         # Top-journal hard case: stop/start trajectories
    seed: int = 42
    background_image_path: Optional[str] = None

    def validate(self) -> None:
        if self.height < 256 or self.width < 256:
            raise ValueError(
                f"Simulation resolution ({self.height}x{self.width}) must be >= 256x256 "
                "to avoid the legacy 96x64 domain-collapse regime."
            )
        if self.min_targets < 0 or self.max_targets < self.min_targets:
            raise ValueError("Invalid target count range.")
        if self.num_frames < 2:
            raise ValueError("num_frames must be >= 2 for temporal Video-SAR modeling.")


@dataclass
class DatasetConfig:
    """Real Video-SAR benchmark dataset & sequence-level splitting config."""

    name: Literal["vsmb", "snl", "synthetic"] = "vsmb"
    root_dir: str = "data/vsmb"
    window_size: int = 5                     # Causal temporal window length (W frames)
    window_stride: int = 1
    tile_size: int = 512                     # Overlapping tile size (512x512)
    tile_overlap: int = 64
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42
    snl_train_end_frame: int = 600           # Standard SNL split: 1-600 train, 601-900 test
    snl_guard_gap_frames: int = 16           # Temporal guard gap to prevent window leakage
    normalize_db: bool = True
    return_tensors: Literal["numpy", "pt"] = "numpy"

    def validate(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}.")
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1.")
        if self.tile_overlap >= self.tile_size:
            raise ValueError("tile_overlap must be strictly smaller than tile_size.")


@dataclass
class EvalConfig:
    """Object-level detection & multi-object tracking evaluation config."""

    iou_threshold: float = 0.5
    small_target_iou_threshold: float = 0.3
    froc_fa_points: Tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
    hota_iou_thresholds: Tuple[float, ...] = (
        0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95
    )
    allow_mixed_synthetic_real: bool = False  # Hard guard: must stay False


@dataclass
class VideoSARConfig:
    """Top-level configuration container."""

    experiment_name: str = "videosar_v2_baseline"
    seed: int = 42
    output_dir: str = "outputs/experiments"
    radar: RadarConfig = field(default_factory=RadarConfig)
    mwp: MWPConfig = field(default_factory=MWPConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    def validate(self) -> None:
        self.simulation.validate()
        self.dataset.validate()
        if self.evaluation.allow_mixed_synthetic_real:
            raise ValueError(
                "Mixing synthetic and real splits in a single evaluation metric is "
                "prohibited. Keep evaluation.allow_mixed_synthetic_real=False."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        if out_path.suffix.lower() in {".yaml", ".yml"}:
            out_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        else:
            out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> "VideoSARConfig":
        src = Path(path)
        raw = (
            yaml.safe_load(src.read_text(encoding="utf-8"))
            if src.suffix.lower() in {".yaml", ".yml"}
            else json.loads(src.read_text(encoding="utf-8"))
        )
        cfg = cls(
            experiment_name=raw.get("experiment_name", "videosar_v2_baseline"),
            seed=raw.get("seed", 42),
            output_dir=raw.get("output_dir", "outputs/experiments"),
            radar=RadarConfig(**raw.get("radar", {})),
            mwp=MWPConfig(**raw.get("mwp", {})),
            simulation=SimulationConfig(**raw.get("simulation", {})),
            dataset=DatasetConfig(**raw.get("dataset", {})),
            evaluation=EvalConfig(**raw.get("evaluation", {})),
        )
        cfg.validate()
        return cfg
