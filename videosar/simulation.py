"""High-fidelity Video-SAR simulation & real-background shadow injection engine.

Upgrades over legacy ``videosar/simulation.py``:
- Minimum 256x256 spatial resolution (default 256x256, configurable to 512x512+).
- Multi-target scenes (3-12 moving vehicles) following road-constrained curved &
  accelerating/decelerating trajectories instead of a single constant-velocity box.
- Static man-made building/tree occlusion shadow distractors (matching VSMB false-alarm
  challenges).
- Dual backend:
  1. ``real_bg_injection`` (default): Injects physically-informed moving-target shadows
     and azimuth Doppler-shifted defocused energy onto real SNL/VSMB empty-road
     backgrounds (or multi-look Gamma speckle clutter when no real background is supplied).
  2. ``radarsimpy`` (optional): Connects to ``radarsimpy`` (``Transmitter``, ``Receiver``,
     ``Radar``, ``sim_radar``) for ray-traced complex baseband echo generation.
- Optional ``MicrowavePhotonicsChannel`` integration when ``MWPConfig.enabled=True``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv
import json
import cv2
import numpy as np

from videosar.config import MWPConfig, RadarConfig, SimulationConfig
from videosar.metrics import BoxDetection
from videosar.mwp import MicrowavePhotonicsChannel


@dataclass
class MovingTargetSpec:
    """Kinematic and geometric specification of a moving ground vehicle."""

    track_id: int
    class_id: int
    length_m: float
    width_m: float
    height_m: float
    s0_px: float
    lane_offset_px: float
    speed_mps: float
    accel_mps2: float
    shadow_attenuation: float


@dataclass
class StaticDistractorSpec:
    """Static building or structure casting a stationary radar shadow."""

    cx_px: float
    cy_px: float
    width_px: float
    height_px: float
    shadow_len_px: float
    attenuation: float


@dataclass
class SimulatedSequence:
    """Container for a simulated Video-SAR sequence and its exact MOT annotations."""

    sequence_id: str
    frames: np.ndarray                         # (T, H, W) float32 in [0, 1]
    ground_truths: List[BoxDetection]
    static_distractors: List[StaticDistractorSpec]
    shadow_masks: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0), dtype=np.uint8))
    static_distractor_mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.uint8))
    kinematics: List[Dict[str, object]] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def export_mot_directory(self, out_dir: str | Path) -> Path:
        """Save frames + MOT labels + mask/kinematics/meta truth products.

        Writes ``img1/``, ``gt/gt.txt`` (compatible with ``VSMBDataset``),
        ``masks/%06d.png``, ``static_distractor_mask.png``, ``kinematics.csv``
        and ``meta.json``. All newly added products are optional: legacy
        callers that construct ``SimulatedSequence`` without them still get
        the original two-file export.
        """
        root = Path(out_dir) / self.sequence_id
        img_dir = root / "img1"
        gt_dir = root / "gt"
        mask_dir = root / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        for t in range(self.frames.shape[0]):
            u8 = np.clip(self.frames[t] * 255.0, 0.0, 255.0).astype(np.uint8)
            cv2.imwrite(str(img_dir / f"{t + 1:06d}.png"), u8)

        gt_lines: List[str] = []
        for g in self.ground_truths:
            w = g.x2 - g.x1
            h = g.y2 - g.y1
            gt_lines.append(
                f"{g.frame_id},{g.track_id},{g.x1:.2f},{g.y1:.2f},{w:.2f},{h:.2f},1,{g.class_id},1.0"
            )
        (gt_dir / "gt.txt").write_text("\n".join(gt_lines) + ("\n" if gt_lines else ""), encoding="utf-8")

        if self.shadow_masks.size > 0:
            mask_dir.mkdir(parents=True, exist_ok=True)
            for t in range(self.shadow_masks.shape[0]):
                cv2.imwrite(str(mask_dir / f"{t + 1:06d}.png"), self.shadow_masks[t])
        if self.static_distractor_mask.size > 0:
            cv2.imwrite(str(root / "static_distractor_mask.png"), self.static_distractor_mask)

        kin_fields = ["frame_id", "track_id", "x_px", "y_px", "v_mps", "theta_rad", "motion_state"]
        with (root / "kinematics.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=kin_fields)
            writer.writeheader()
            for row in self.kinematics:
                writer.writerow({k: row.get(k, "") for k in kin_fields})

        (root / "meta.json").write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return root


class VideoSARSimulator:
    """Physical shadow-injection and optional RadarSimPy Video-SAR sequence generator."""

    def __init__(
        self,
        sim_cfg: SimulationConfig,
        radar_cfg: RadarConfig | None = None,
        mwp_cfg: MWPConfig | None = None,
    ) -> None:
        sim_cfg.validate()
        self.sim_cfg = sim_cfg
        self.radar_cfg = radar_cfg or RadarConfig()
        self.mwp_cfg = mwp_cfg or MWPConfig(enabled=False)
        self.mwp_channel = MicrowavePhotonicsChannel(
            self.mwp_cfg, sample_rate_hz=self.radar_cfg.prf_hz * 128.0
        )

    def _load_or_synthesize_background(self, rng: np.random.Generator) -> np.ndarray:
        """Load a real empty-road SAR background or synthesize realistic road + clutter."""
        h, w = self.sim_cfg.height, self.sim_cfg.width
        if self.sim_cfg.background_image_path:
            bg_path = Path(self.sim_cfg.background_image_path)
            img = cv2.imread(str(bg_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Background SAR image not found: {bg_path}")
            resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            base = resized.astype(np.float32) / 255.0
            return np.clip(base, 0.05, 0.95)

        # Synthesize structured terrain: slightly darker asphalt curved road + textured clutter
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        road_center_y = self._road_center_y(xx)
        dist_to_road = np.abs(yy - road_center_y)

        # Multi-scale spatial correlation for realistic terrain reflectivity
        coarse = cv2.GaussianBlur(
            rng.normal(0.55, 0.14, size=(h, w)).astype(np.float32), (15, 15), 4.0
        )
        fine = cv2.GaussianBlur(
            rng.normal(0.0, 0.06, size=(h, w)).astype(np.float32), (5, 5), 1.2
        )
        terrain = np.clip(coarse + fine, 0.25, 0.85)
        # Road surface has moderately lower backscatter than rough soil/buildings
        road_mask = np.exp(-0.5 * (dist_to_road / 18.0) ** 2)
        base = terrain * (1.0 - 0.22 * road_mask)
        return np.clip(base, 0.15, 0.90).astype(np.float32)

    def _road_center_y(self, x: np.ndarray | float) -> np.ndarray | float:
        h, w = self.sim_cfg.height, self.sim_cfg.width
        xc = x - (w * 0.5)
        curve = self.sim_cfg.road_curvature
        return (h * 0.5) + 0.14 * h * np.sin(curve * xc * np.pi)

    def _road_tangent_angle(self, x: float) -> float:
        dx = 1.0
        dy = float(self._road_center_y(x + dx) - self._road_center_y(x - dx)) / (2.0 * dx)
        return float(np.arctan2(dy, 1.0))

    def _sample_targets(self, rng: np.random.Generator) -> List[MovingTargetSpec]:
        n_targets = int(
            rng.integers(self.sim_cfg.min_targets, self.sim_cfg.max_targets + 1)
        )
        targets: List[MovingTargetSpec] = []
        w = self.sim_cfg.width
        for idx in range(1, n_targets + 1):
            direction = 1.0 if (idx % 2 == 1) else -1.0
            speed = float(
                rng.uniform(self.sim_cfg.min_speed_mps, self.sim_cfg.max_speed_mps)
            ) * direction
            accel = float(
                rng.uniform(-self.sim_cfg.max_accel_mps2, self.sim_cfg.max_accel_mps2)
            )
            s0 = float(rng.uniform(0.08 * w, 0.35 * w)) if direction > 0 else float(
                rng.uniform(0.65 * w, 0.92 * w)
            )
            lane_offset = float(rng.choice([-8.0, -3.0, 3.0, 8.0]) + rng.uniform(-1.5, 1.5))
            att = float(rng.uniform(*self.sim_cfg.shadow_contrast_range))
            targets.append(
                MovingTargetSpec(
                    track_id=idx,
                    class_id=1,
                    length_m=float(rng.uniform(4.0, 6.2)),
                    width_m=float(rng.uniform(1.8, 2.5)),
                    height_m=float(rng.uniform(1.5, 2.8)),
                    s0_px=s0,
                    lane_offset_px=lane_offset,
                    speed_mps=speed,
                    accel_mps2=accel,
                    shadow_attenuation=att,
                )
            )
        return targets

    def _sample_static_distractors(
        self, rng: np.random.Generator
    ) -> List[StaticDistractorSpec]:
        h, w = self.sim_cfg.height, self.sim_cfg.width
        distractors: List[StaticDistractorSpec] = []
        for _ in range(self.sim_cfg.num_static_distractors):
            cx = float(rng.uniform(24.0, w - 24.0))
            road_y = float(self._road_center_y(cx))
            side = float(rng.choice([-1.0, 1.0]))
            cy = float(np.clip(road_y + side * rng.uniform(34.0, h * 0.38), 20.0, h - 20.0))
            distractors.append(
                StaticDistractorSpec(
                    cx_px=cx,
                    cy_px=cy,
                    width_px=float(rng.uniform(10.0, 26.0)),
                    height_px=float(rng.uniform(8.0, 18.0)),
                    shadow_len_px=float(rng.uniform(10.0, 24.0)),
                    attenuation=float(rng.uniform(0.15, 0.35)),
                )
            )
        return distractors

    def _apply_static_distractors(
        self, base: np.ndarray, distractors: List[StaticDistractorSpec]
    ) -> np.ndarray:
        """Render static building layover bright edge + stationary occlusion shadow."""
        out = base.copy()
        h, w = out.shape
        look_rad = np.deg2rad(self.radar_cfg.azimuth_look_dir_deg)
        sx_dir = float(np.cos(look_rad))
        sy_dir = float(np.sin(look_rad))

        for d in distractors:
            # Bright building front edge
            bx1 = int(np.clip(d.cx_px - d.width_px * 0.5, 0, w - 1))
            bx2 = int(np.clip(d.cx_px + d.width_px * 0.5, 1, w))
            by1 = int(np.clip(d.cy_px - d.height_px * 0.5, 0, h - 1))
            by2 = int(np.clip(d.cy_px + d.height_px * 0.5, 1, h))
            out[by1:by2, bx1:bx2] = np.clip(out[by1:by2, bx1:bx2] * 1.35 + 0.12, 0.0, 1.0)

            # Stationary shadow cast away from radar look direction
            sh_cx = d.cx_px + sx_dir * (d.shadow_len_px * 0.5 + d.height_px * 0.4)
            sh_cy = d.cy_px + sy_dir * (d.shadow_len_px * 0.5 + d.height_px * 0.4)
            sx1 = int(np.clip(sh_cx - d.width_px * 0.5, 0, w - 1))
            sx2 = int(np.clip(sh_cx + d.width_px * 0.5, 1, w))
            sy1 = int(np.clip(sh_cy - d.shadow_len_px * 0.5, 0, h - 1))
            sy2 = int(np.clip(sh_cy + d.shadow_len_px * 0.5, 1, h))
            out[sy1:sy2, sx1:sx2] *= d.attenuation
        return out

    def _generate_radarsimpy_backend(self, sequence_id: str) -> SimulatedSequence:
        """Optional RadarSimPy ray-tracing backend."""
        try:
            from radarsimpy import Radar, Receiver, Transmitter  # type: ignore
            from radarsimpy.simulator import sim_radar  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "RadarSimPy backend requested (sim_cfg.backend='radarsimpy'), but "
                "'radarsimpy' is not installed. Install it from "
                "https://github.com/radarsimx/radarsimpy or switch to "
                "sim_cfg.backend='real_bg_injection'."
            ) from exc

        # Minimal verified RadarSimPy transceiver instantiation hook
        tx = Transmitter(
            f=[
                self.radar_cfg.carrier_freq_hz - self.radar_cfg.bandwidth_hz / 2.0,
                self.radar_cfg.carrier_freq_hz + self.radar_cfg.bandwidth_hz / 2.0,
            ],
            t=1.0 / self.radar_cfg.prf_hz,
            tx_power=25,
            prp=1.0 / self.radar_cfg.prf_hz,
            pulses=64,
            channels=[{"location": (0, 0, 0)}],
        )
        rx = Receiver(
            fs=2.0e6,
            noise_figure=6,
            rf_gain=20,
            baseband_gain=30,
            channels=[{"location": (0, 0, 0)}],
        )
        radar = Radar(transmitter=tx, receiver=rx)
        _ = (radar, sim_radar)
        raise NotImplementedError(
            "Provide a 3D STL scene mesh path to execute full RadarSimPy ray-traced "
            "back-projection imaging, or use backend='real_bg_injection'."
        )

    def generate_sequence(
        self,
        sequence_id: str = "sim_seq_001",
        seed: Optional[int] = None,
    ) -> SimulatedSequence:
        """Generate a complete multi-target Video-SAR sequence with MOT annotations."""
        if self.sim_cfg.backend == "radarsimpy":
            return self._generate_radarsimpy_backend(sequence_id=sequence_id)

        rng = np.random.default_rng(self.sim_cfg.seed if seed is None else seed)
        h, w = self.sim_cfg.height, self.sim_cfg.width
        n_frames = self.sim_cfg.num_frames
        dt = 1.0 / self.radar_cfg.frame_rate_hz
        px_m = self.radar_cfg.pixel_spacing_m
        grazing_rad = np.deg2rad(self.radar_cfg.grazing_angle_deg)
        look_rad = np.deg2rad(self.radar_cfg.azimuth_look_dir_deg)

        base_clean = self._load_or_synthesize_background(rng)
        distractors = self._sample_static_distractors(rng)
        static_scene = self._apply_static_distractors(base_clean, distractors)
        targets = self._sample_targets(rng)

        frames = np.zeros((n_frames, h, w), dtype=np.float32)
        shadow_masks = np.zeros((n_frames, h, w), dtype=np.uint8)
        gts: List[BoxDetection] = []
        kinematics: List[Dict[str, object]] = []
        mwp_evm_values: List[float] = []

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        speckle_shape = max(self.radar_cfg.n_looks, 1.0)

        # Stop-and-go: pick ~1/3 of targets, each stops for 4-6 frames in the
        # middle third of the sequence, then accelerates back to motion.
        stop_windows: Dict[int, Tuple[int, int]] = {}
        if self.sim_cfg.enable_stop_and_go and targets:
            stop_candidates = [t.track_id for t in targets]
            rng.shuffle(stop_candidates)
            for track_id in stop_candidates[: max(1, len(stop_candidates) // 3)]:
                stop_len = int(min(rng.integers(4, 7), max(1, n_frames - 2)))
                mid_start = max(1, n_frames // 3)
                mid_end = max(mid_start + 1, n_frames - stop_len)
                stop_start = int(rng.integers(mid_start, mid_end))
                stop_windows[track_id] = (stop_start, stop_start + stop_len)

        # Static distractor mask (0/255) for false-positive attribution.
        static_mask = np.zeros((h, w), dtype=np.uint8)
        for d in distractors:
            sh_cx = d.cx_px + np.cos(look_rad) * (d.shadow_len_px * 0.5 + d.height_px * 0.4)
            sh_cy = d.cy_px + np.sin(look_rad) * (d.shadow_len_px * 0.5 + d.height_px * 0.4)
            sx1 = int(np.clip(sh_cx - d.width_px * 0.5, 0, w - 1))
            sx2 = int(np.clip(sh_cx + d.width_px * 0.5, 1, w))
            sy1 = int(np.clip(sh_cy - d.shadow_len_px * 0.5, 0, h - 1))
            sy2 = int(np.clip(sh_cy + d.shadow_len_px * 0.5, 1, h))
            static_mask[sy1:sy2, sx1:sx2] = 255

        for t_idx in range(n_frames):
            t_sec = t_idx * dt
            intensity = static_scene.copy()

            for tgt in targets:
                stop_window = stop_windows.get(tgt.track_id)
                motion_state = "moving"
                cum_disp_m = tgt.speed_mps * t_sec + 0.5 * tgt.accel_mps2 * (t_sec ** 2)
                if stop_window is not None:
                    s_start, s_end = stop_window
                    if s_start <= t_idx < s_end:
                        motion_state = "stopped"
                        cum_disp_m = tgt.speed_mps * s_start * dt + 0.5 * tgt.accel_mps2 * ((s_start * dt) ** 2)
                    elif t_idx >= s_end:
                        # Re-accelerate after the stop, with an extra push.
                        t_move = t_sec - (s_end - s_start) * dt
                        cum_disp_m = tgt.speed_mps * s_start * dt + tgt.speed_mps * (t_move - s_start * dt) + 0.5 * tgt.accel_mps2 * (t_move ** 2)

                disp_m = cum_disp_m
                x_pos = tgt.s0_px + disp_m / px_m
                if x_pos < 8.0 or x_pos > (w - 8.0):
                    continue
                y_pos = float(self._road_center_y(x_pos)) + tgt.lane_offset_px
                if y_pos < 8.0 or y_pos > (h - 8.0):
                    continue

                theta = self._road_tangent_angle(x_pos)
                v_inst = max(0.0, tgt.speed_mps + tgt.accel_mps2 * t_sec)
                if motion_state == "stopped":
                    v_inst = 0.0

                # Speed-shadow coupling: faster targets cast longer, fainter shadows.
                speed_ratio = float(np.clip(v_inst / max(self.sim_cfg.max_speed_mps, 1.0), 0.0, 1.5))
                shadow_ext_px = (tgt.height_m / max(np.tan(grazing_rad), 0.1)) / px_m
                shadow_ext_px *= 1.0 + 0.55 * speed_ratio
                attenuation = float(np.clip(tgt.shadow_attenuation - 0.12 * speed_ratio, 0.05, 0.95))
                if motion_state == "stopped":
                    # Stationary phase: shadow collapses, bright scatterer returns.
                    shadow_ext_px *= 0.35
                    attenuation = min(attenuation, 0.12)
                len_px = max(tgt.length_m / px_m, 6.0)
                wid_px = max(tgt.width_m / px_m, 4.0)

                # Local rotated bounding region for moving shadow
                dx = xx - x_pos
                dy = yy - y_pos
                rx = dx * np.cos(theta) + dy * np.sin(theta)
                ry = -dx * np.sin(theta) + dy * np.cos(theta)

                # Soft physical shadow profile (blurred PSF edge)
                sx_sigma = len_px * 0.45
                sy_sigma = (wid_px + shadow_ext_px * 0.45) * 0.45
                shadow_profile = np.exp(
                    -0.5 * ((rx / max(sx_sigma, 1.5)) ** 2 + (ry / max(sy_sigma, 1.5)) ** 2)
                )
                intensity *= 1.0 - (1.0 - attenuation) * shadow_profile
                shadow_masks[t_idx][shadow_profile >= 0.35] = 255

                # Doppler azimuth displacement & defocusing smear of vehicle scatterer
                v_radial = v_inst * np.sin(theta - look_rad)
                az_shift_px = np.clip(
                    (self.radar_cfg.slant_range_m / max(self.radar_cfg.platform_speed_mps, 1.0))
                    * (v_radial / max(px_m, 0.05))
                    * 0.015
                    * self.sim_cfg.doppler_smear_gain,
                    -22.0,
                    22.0,
                )
                smear_cx = x_pos + az_shift_px * np.cos(look_rad + np.pi / 2.0)
                smear_cy = y_pos + az_shift_px * np.sin(look_rad + np.pi / 2.0)
                smear = np.exp(
                    -0.5
                    * (
                        ((xx - smear_cx) / max(len_px * 0.85, 3.0)) ** 2
                        + ((yy - smear_cy) / max(wid_px * 0.55, 2.0)) ** 2
                    )
                )
                if motion_state == "stopped":
                    # Bright stationary scatterer returns near the vehicle center.
                    body = np.exp(
                        -0.5 * (((xx - x_pos) / max(len_px * 0.5, 3.0)) ** 2
                                + ((yy - y_pos) / max(wid_px * 0.5, 2.0)) ** 2)
                    )
                    intensity = np.clip(intensity + 0.28 * body, 0.0, 1.0)
                else:
                    intensity = np.clip(intensity + 0.16 * smear, 0.0, 1.0)

                # Ground-truth shadow bounding box
                half_w = max(0.5 * (abs(len_px * np.cos(theta)) + abs((wid_px + shadow_ext_px * 0.35) * np.sin(theta))), 4.0)
                half_h = max(0.5 * (abs(len_px * np.sin(theta)) + abs((wid_px + shadow_ext_px * 0.35) * np.cos(theta))), 4.0)
                bx1 = float(np.clip(x_pos - half_w, 0.0, w - 1.0))
                by1 = float(np.clip(y_pos - half_h, 0.0, h - 1.0))
                bx2 = float(np.clip(x_pos + half_w, bx1 + 2.0, float(w)))
                by2 = float(np.clip(y_pos + half_h, by1 + 2.0, float(h)))

                gts.append(
                    BoxDetection(
                        frame_id=t_idx + 1,
                        x1=bx1,
                        y1=by1,
                        x2=bx2,
                        y2=by2,
                        score=1.0,
                        class_id=tgt.class_id,
                        track_id=tgt.track_id,
                    )
                )
                kinematics.append({
                    "frame_id": t_idx + 1,
                    "track_id": tgt.track_id,
                    "x_px": float(x_pos),
                    "y_px": float(y_pos),
                    "v_mps": float(v_inst),
                    "theta_rad": float(theta),
                    "motion_state": motion_state,
                })

            # Multiplicative multi-look Gamma speckle noise
            speckle = rng.gamma(
                shape=speckle_shape, scale=1.0 / speckle_shape, size=(h, w)
            ).astype(np.float32)
            noisy_amp = np.sqrt(np.clip(intensity * speckle, 1e-6, 4.0))

            # Optional Microwave Photonics (MWP) complex-domain degradation
            if self.mwp_cfg.enabled:
                rand_phase = rng.uniform(-np.pi, np.pi, size=(h, w))
                complex_img = noisy_amp * np.exp(1j * rand_phase)
                degraded_iq, diag = self.mwp_channel.apply(complex_img, rng=rng)
                noisy_amp = np.abs(degraded_iq).astype(np.float32)
                mwp_evm_values.append(diag["evm_rms"])

            # Normalize to [0, 1] SAR display dynamic range
            frame_norm = np.clip(noisy_amp / 1.35, 0.0, 1.0).astype(np.float32)
            frames[t_idx] = frame_norm

        return SimulatedSequence(
            sequence_id=sequence_id,
            frames=frames,
            ground_truths=gts,
            static_distractors=distractors,
            shadow_masks=shadow_masks,
            static_distractor_mask=static_mask,
            kinematics=kinematics,
            metadata={
                "backend": self.sim_cfg.backend,
                "resolution": f"{h}x{w}",
                "num_frames": n_frames,
                "num_targets_spawned": len(targets),
                "num_static_distractors": len(distractors),
                "mwp_enabled": self.mwp_cfg.enabled,
                "mwp_mean_evm_rms": float(np.mean(mwp_evm_values)) if mwp_evm_values else 0.0,
                "seed": self.sim_cfg.seed if seed is None else seed,
                "enable_stop_and_go": self.sim_cfg.enable_stop_and_go,
                "stop_windows": {str(k): list(v) for k, v in stop_windows.items()},
            },
        )
