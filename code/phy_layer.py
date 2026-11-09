# phy_layer.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Time and rate constants used across the codebase
SLOT_DURATION_S = 1e-3           # 1 ms slot
CHANNEL_RATE_BPS = 100_000_000   # nominal PHY (before SINR scaling)
THROUGHPUT_SCALE = 20.0         # scales bits per slot without changing utilization


@dataclass
class PHYCfg:
    sinr_mu_db: float = 12.0       # mean large-scale SINR (dB)
    sinr_sigma_db: float = 4.5     # log-normal spread (dB)
    rayleigh_scale: float = 1.0    # small-scale fading parameter
    shadow_sigma_db: float = 4.0   # log-normal shadowing std (dB)
    interf_db_mu: float = -6.0     # mean interference in dB relative to signal
    interf_db_sigma: float = 3.0   # spread of interference (dB)


class PHYLayer:
    """
    Lightweight PHY helper used by the environment.
    Provides:
      - slot_bits(sinr_db): capacity in bits for a SLOT_DURATION_S slot
      - energy_joules(tx_power_dbm, duration_s, circuit_power_w): TX energy consumption
      - sample_sinr_db(...): draw a SINR sample (backward compatible with older kwargs)
    """
    cfg = PHYCfg()

    @staticmethod
    def set_cfg(**kwargs) -> None:
        """Optionally adjust default PHY parameters at runtime."""
        for k, v in kwargs.items():
            if hasattr(PHYLayer.cfg, k):
                setattr(PHYLayer.cfg, k, v)

    # ---------- core mappings ----------
    @staticmethod
    def _db_to_linear(db: float) -> float:
        return float(10.0 ** (float(db) / 10.0))

    @staticmethod
    def _linear_to_db(x: float) -> float:
        x = max(float(x), 1e-12)
        return float(10.0 * np.log10(x))

    # Shannon-like mapping with a conservative spectral efficiency cap
    @staticmethod
    def _spectral_efficiency_bps_per_hz(sinr_linear: float) -> float:
        se = float(np.log2(1.0 + max(float(sinr_linear), 1e-12)))
        return float(min(se, 6.0))  # cap at ~64-QAM-ish bound

    # ---------- capacity / energy ----------
    @staticmethod
    def slot_bits(sinr_db: float, bw_hz: float = 1e6) -> int:
        """
        Compute the bits that can be sent in one slot at given SINR.
        By default, bandwidth is 1 MHz.
        """
        sinr_lin = PHYLayer._db_to_linear(sinr_db)
        se = PHYLayer._spectral_efficiency_bps_per_hz(sinr_lin)
        bits = se * float(bw_hz) * float(SLOT_DURATION_S) * float(THROUGHPUT_SCALE)

        if not np.isfinite(bits) or bits < 0.0:
            bits = 0.0
        return int(max(0, round(bits)))

    @staticmethod
    def energy_joules(tx_power_dbm: float, duration_s: float, circuit_power_w: float = 0.0) -> float:
        """
        Energy = (P_tx + P_circuit) * duration.
        P_tx uses dBm→W conversion.
        """
        p_w = 10.0 ** ((float(tx_power_dbm) - 30.0) / 10.0)
        e = (p_w + float(circuit_power_w)) * float(duration_s)
        return float(max(e, 0.0))

    @staticmethod
    def required_sinr_db_for_bits(bits: int, bw_hz: float = 1e6) -> float:
        """
        Invert the Shannon-like mapping to estimate required SINR for given bits/slot.
        Used as a simple outage test; conservative due to SE cap.
        """
        bits = int(max(0, bits))
        if bits == 0:
            return -20.0

        Ts = float(SLOT_DURATION_S)
        denom = max(float(bw_hz) * Ts * float(THROUGHPUT_SCALE), 1e-12)
        se_req = float(bits) / denom
        se_req = float(np.clip(se_req, 1e-6, 6.0))

        sinr_lin = (2.0 ** se_req) - 1.0
        sinr_db = PHYLayer._linear_to_db(sinr_lin)
        return float(np.clip(sinr_db, -20.0, 40.0))

    @staticmethod
    def sample_link_sinr_db(
        rng: np.random.Generator, *, base_mu_db: float, age_frac: float, drag: float
    ) -> float:
        """
        Simple policy-linked channel stressor:
        higher age (queueing) and higher drag increase outage/interruption risk.
        """
        mu = float(base_mu_db) - 5.0 * float(drag) - 4.0 * float(np.clip(age_frac, 0.0, 1.0))
        return PHYLayer.sample_sinr_db(
            rng,
            base_sinr_db=mu,
            sigma_db=float(PHYLayer.cfg.sinr_sigma_db),
            enable_fading=True,
            enable_shadowing=True,
            enable_interference=True,
        )

    @staticmethod
    def mcs_multiplier_from_sinr_db(sinr_db: float) -> float:
        """
        Returns a multiplier applied to a base rate (channel_rate_bps).
        Coarse but monotone and stable.
        """
        x = float(sinr_db)
        if x < -2.0:
            return 0.0
        if x < 4.0:
            return 0.5
        if x < 10.0:
            return 1.0
        if x < 16.0:
            return 2.0
        return 3.5

    # ---------- SINR sampling / link adaptation ----------
    @staticmethod
    def sample_sinr_db(
            rng: np.random.Generator,
            base_sinr_db: float | None = None,
            sigma_db: float | None = None,
            **kwargs,
    ) -> float:
        """
        Sample an instantaneous SINR in dB.

        Backward compatible:
          - accepts legacy keyword arguments: base_mu_db, mu_db, mean_db
          - accepts optional feature flags (enable_fading/shadowing/interference) via kwargs (ignored here)
        """
        if base_sinr_db is None:
            if "base_mu_db" in kwargs:
                base_sinr_db = kwargs["base_mu_db"]
            elif "mu_db" in kwargs:
                base_sinr_db = kwargs["mu_db"]
            elif "mean_db" in kwargs:
                base_sinr_db = kwargs["mean_db"]
            else:
                base_sinr_db = float(getattr(PHYLayer.cfg, "sinr_mu_db", 12.0))

        if sigma_db is None:
            sigma_db = kwargs.get("sigma_db", None)
        if sigma_db is None:
            sigma_db = float(getattr(PHYLayer.cfg, "sinr_sigma_db", 6.0))

        # log-normal-ish fast fading in dB
        return float(float(base_sinr_db) + rng.normal(0.0, float(sigma_db)))

    @staticmethod
    def rate_multiplier_from_sinr_db(sinr_db: float) -> float:
        """
        Simple SINR→rate mapping (piecewise, monotone).
        Returns a multiplier on nominal bits-per-sample.
        This is intentionally simple but produces clear separations.
        """
        s = float(sinr_db)
        if s < -2.0:
            return 0.05
        if s < 2.0:
            return 0.15
        if s < 6.0:
            return 0.35
        if s < 10.0:
            return 0.65
        if s < 14.0:
            return 1.00
        if s < 18.0:
            return 1.25
        return 1.45

    @staticmethod
    def power_control_boost_db(
            sinr_db: float, target_sinr_db: float, alpha: float, boost_max_db: float
    ) -> float:
        """
        Mild power-control: if SINR below target, increase power a bit (clipped).
        """
        shortfall = float(target_sinr_db) - float(sinr_db)
        if shortfall <= 0.0:
            return 0.0
        boost = float(alpha) * shortfall
        return float(np.clip(boost, 0.0, float(boost_max_db)))