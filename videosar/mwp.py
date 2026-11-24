"""Microwave Photonics (MWP) RF-optical link degradation module.

Explicitly models the three primary physical impairments of a photonic-assisted
Video-SAR transceiver when ``MWPConfig.enabled=True``:
1. Laser phase noise (Wiener process with Lorentzian linewidth ``delta_nu``).
2. Electro-optic Mach-Zehnder modulator (MZM) transfer nonlinearity (sinusoidal
   compression & third-order intermodulation around quadrature bias).
3. Photodetector shot noise + receiver thermal noise.
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from videosar.config import MWPConfig

ELEMENTARY_CHARGE_C = 1.602176634e-19


class MicrowavePhotonicsChannel:
    """Applies photonic transceiver impairments to complex baseband SAR echoes."""

    def __init__(self, config: MWPConfig, sample_rate_hz: float = 1.0e6) -> None:
        self.config = config
        self.sample_rate_hz = float(sample_rate_hz)

    def apply(
        self,
        iq_signal: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Pass complex IQ echo array through the MWP link model.

        Args:
            iq_signal: Complex ndarray of shape ``(..., N_samples)``.
            rng: Optional NumPy random generator for reproducibility.

        Returns:
            ``(degraded_iq, diagnostics)`` where diagnostics reports EVM and SNR impact.
        """
        arr = np.asarray(iq_signal, dtype=np.complex128)
        if not self.config.enabled:
            return arr.copy(), {
                "mwp_enabled": 0.0,
                "evm_rms": 0.0,
                "phase_noise_std_rad": 0.0,
            }

        gen = rng if rng is not None else np.random.default_rng(0)
        clean_ref = arr.copy()
        out = arr.copy()

        # 1. Electro-optic MZM nonlinearity (quadrature bias + bias drift)
        if self.config.include_mzm_nonlinearity:
            scale = np.max(np.abs(out)) + 1e-12
            norm_re = np.real(out) / scale
            norm_im = np.imag(out) / scale
            m = self.config.mzm_modulation_index
            drift = self.config.mzm_bias_drift_rad
            # sin(m*x + drift) - sin(drift) linearized by dividing by m*cos(drift)
            gain = max(m * np.cos(drift), 1e-6)
            nl_re = (np.sin(m * norm_re + drift) - np.sin(drift)) / gain
            nl_im = (np.sin(m * norm_im + drift) - np.sin(drift)) / gain
            out = (nl_re + 1j * nl_im) * scale

        # 2. Laser phase noise (Wiener process: var(dphi) = 2 * pi * delta_nu * dt)
        phase_std = 0.0
        if self.config.include_phase_noise and self.config.laser_linewidth_hz > 0:
            dt = 1.0 / max(self.sample_rate_hz, 1.0)
            step_std = np.sqrt(2.0 * np.pi * self.config.laser_linewidth_hz * dt)
            dphi = gen.normal(0.0, step_std, size=out.shape)
            phi = np.cumsum(dphi, axis=-1)
            phase_std = float(np.std(phi))
            out = out * np.exp(1j * phi)

        # 3. Photodetector shot noise + thermal noise
        if self.config.include_shot_noise:
            i_dc = self.config.pd_responsivity_a_per_w * (self.config.optical_power_mw * 1e-3)
            # Normalized relative shot noise amplitude
            shot_var = 2.0 * ELEMENTARY_CHARGE_C * max(i_dc, 1e-9) * self.sample_rate_hz
            shot_std = np.sqrt(shot_var) / max(i_dc, 1e-9)
            sig_rms = np.sqrt(np.mean(np.abs(out) ** 2)) + 1e-12
            noise_std = np.hypot(shot_std, self.config.thermal_noise_std) * sig_rms
            noise = (
                gen.normal(0.0, noise_std / np.sqrt(2.0), size=out.shape)
                + 1j * gen.normal(0.0, noise_std / np.sqrt(2.0), size=out.shape)
            )
            out = out + noise

        ref_rms = float(np.sqrt(np.mean(np.abs(clean_ref) ** 2)) + 1e-12)
        err_rms = float(np.sqrt(np.mean(np.abs(out - clean_ref) ** 2)))
        diagnostics = {
            "mwp_enabled": 1.0,
            "evm_rms": err_rms / ref_rms,
            "phase_noise_std_rad": phase_std,
        }
        return out, diagnostics
