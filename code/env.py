from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
from collections import deque

from phy_layer import PHYLayer, SLOT_DURATION_S, CHANNEL_RATE_BPS


# ============================================================
# Device profiles
# ============================================================

@dataclass
class DeviceProfile:
    name: str
    tx_power_dbm: float
    sampling_hz: float
    bits_per_sample: int
    battery_joules: float
    energy_per_bit: float = 1.2e-8
    drag_penalty: float = 0.0


DEFAULT_PROFILES: Dict[int, DeviceProfile] = {
    0: DeviceProfile("Emergency", 18.0, 5.0, 4000, 50.0, 1.0e-8, 0.0),
    1: DeviceProfile("Medical",   6.0, 0.2, 2000, 20.0, 0.8e-8, 0.0),
    2: DeviceProfile("Fitness",   0.0, 1.0, 1500, 15.0, 1.8e-8, 0.05),
}


# ============================================================
# Base single-agent environment (physics + queues)
# ============================================================

class SpectrumEnv:
    """
    NOT used directly by trainers.
    Only wrapped by MultiAgentSpectrumEnv.
    """

    def __init__(
            self,
            *,
            steps_per_ep: int,
            n_devices: int,
            n_channels: int,
            seed: int,
            arrival_scale: float,
            class_mix: Tuple[float, float, float],
            burst_k: float | None,
            device_skew: float,
            bits_scale: float,
            delay_penalty_w: float,
            energy_penalty_w: float,
            prio_weights: Tuple[float, float, float],
            device_select_mode: str,
            device_jitter_cv: float = 0.0,
    ) -> None:

        self.rng = np.random.default_rng(seed)
        self.phy_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 88491]))

        self.steps_per_ep = int(steps_per_ep)
        self.n_devices = int(n_devices)
        self.n_channels = int(n_channels)
        self.n_classes = 3

        self.slot_s = float(SLOT_DURATION_S)
        self.channel_rate_bps = float(CHANNEL_RATE_BPS)

        self.device_select_mode = str(device_select_mode)
        self._prio_w = tuple(float(x) for x in prio_weights)
        self._delay_pen_w = float(delay_penalty_w)
        self._energy_pen_w = float(energy_penalty_w)

        # class assignment
        mix = np.asarray(class_mix, float)
        mix = mix / max(mix.sum(), 1e-12)
        self.dev_class = self.rng.choice(3, size=self.n_devices, p=mix)

        # offered-load heterogeneity
        self.dev_load_scale = np.ones(self.n_devices, dtype=float)
        if float(device_skew) > 0:
            raw = self.rng.pareto(1.0 + float(device_skew), size=self.n_devices) + 1.0
            self.dev_load_scale = (raw / max(raw.mean(), 1e-12)).astype(float)

        self.profiles = DEFAULT_PROFILES

        # per-device params
        self.tx_power_dbm = np.zeros(self.n_devices, dtype=float)
        self.sampling_hz = np.zeros(self.n_devices, dtype=float)
        self.bits_per_sample = np.zeros(self.n_devices, dtype=int)
        self.battery_cap_j = np.zeros(self.n_devices, dtype=float)
        self.energy_per_bit = np.zeros(self.n_devices, dtype=float)
        self.drag_penalty = np.zeros(self.n_devices, dtype=float)

        for i in range(self.n_devices):
            p = self.profiles[int(self.dev_class[i])]
            self.tx_power_dbm[i] = float(p.tx_power_dbm)
            self.sampling_hz[i] = float(p.sampling_hz) * float(arrival_scale) * float(self.dev_load_scale[i])
            self.bits_per_sample[i] = int(max(1, round(float(p.bits_per_sample) * float(bits_scale))))
            self.battery_cap_j[i] = float(p.battery_joules)
            self.energy_per_bit[i] = float(p.energy_per_bit)
            self.drag_penalty[i] = float(p.drag_penalty)

        # ------------------------------------------------------------
        # Per-device hardware jitter (energy variability)
        # device_jitter_cv = coefficient of variation
        # Introduces multiplicative log-normal variability in:
        #   - energy_per_bit
        #   - circuit power
        # ------------------------------------------------------------
        self.device_jitter_cv = float(device_jitter_cv)

        if self.device_jitter_cv > 0.0:
            sigma = float(self.device_jitter_cv)
            jitter = np.exp(self.rng.normal(0.0, sigma, size=self.n_devices))
            jitter = jitter / float(np.mean(jitter))  # normalize mean=1
            jitter = np.clip(jitter, 0.5, 1.8)

            self.energy_per_bit *= jitter
            self.tx_power_dbm += 10.0 * np.log10(jitter)
        # Keep these for stress-profile logic and arrivals process
        self.arrival_scale = float(arrival_scale)
        self.burst_k = float(burst_k or 0.0)
        self.device_skew = float(device_skew)

        # Queue cap in bits (defensive). If you later expose a CLI knob, keep this as default.
        # 50 "nominal packets" worth of max bits-per-sample is a reasonable cap for stability.
        self.queue_cap_bits = float(getattr(self, "queue_cap_bits", 50.0 * float(np.max(self.bits_per_sample))))

        # ------------------------------------------------------------
        # Per-device arrival heterogeneity (used in _arrivals_step).
        # A value of 1.0 means nominal traffic.
        # device_skew controls log-normal dispersion; 0.0 keeps all devices identical.
        # ------------------------------------------------------------
        if getattr(self, "device_skew", 0.0) and self.device_skew > 0:
            mult = np.exp(self.rng.normal(loc=0.0, scale=float(self.device_skew), size=self.n_devices))
            mult = mult / float(np.mean(mult))  # normalize mean to 1.0
            self.arrival_mult = np.clip(mult, 0.25, 4.0)
        else:
            self.arrival_mult = np.ones(self.n_devices, dtype=float)
        self.battery_j = self.battery_cap_j.copy()

        # ============================================================
        # Channel heterogeneity (large-scale + episode-stable shadowing)
        # ============================================================
        self.pathloss_db = self.rng.uniform(90.0, 120.0, size=self.n_devices).astype(float)

        # These are re-drawn once per episode in reset() so each episode is stable
        self.shadow_db = self.rng.normal(0.0, 6.0, size=self.n_devices).astype(float)

        self.interf_w_base = float(1e-10)
        phy_noise = getattr(PHYLayer, "NOISE_W", None)
        self.noise_w = float(phy_noise) if phy_noise is not None else 1e-10

        # Per-episode versions (set in reset)
        self.interf_w_ep = self.interf_w_base
        self.shadow_db_ep = self.shadow_db.copy()

        # ============================================================
        # PHY / link-adaptation knobs (minimal, policy-coupled via device choice)
        # ============================================================
        self.sinr_sigma_db = float(getattr(PHYLayer.cfg, "sinr_sigma_db", 3.5))

        # Per-device average SINR baseline (stable given seed; scheduling matters via device selection)
        self.base_sinr_db = self.rng.normal(loc=10.0, scale=4.0, size=self.n_devices).astype(float)

        # Backward-compatible alias: existing step() prefers mean_sinr_db if present
        self.mean_sinr_db = self.base_sinr_db.copy()

        # circuit + sensing energy terms (W)
        self.circuit_power_w = 0.18  # paid for any scheduled attempt (TX chain + baseband)
        self.sense_power_w = 0.06  # paid when action==0 on serviceable backlog (decision/sensing)

        # Backward-compatible alias used by current energy accounting path
        self.tx_circuit_w = float(self.circuit_power_w)

        # Optional baseline (kept for compatibility if you already use it)
        self.idle_circuit_w = float(getattr(self, "idle_circuit_w", 0.02))

        # interruption/outage waste fractions (kept as-is; step() already uses these)
        self.intr_waste_frac = float(getattr(self, "intr_waste_frac", 0.90))
        self.outage_waste_frac = float(getattr(self, "outage_waste_frac", 0.70))

        # optional power-control coupling (kept mild; avoids turning this into a full PHY paper)
        self.enable_power_control = True
        self.target_sinr_db = 8.0
        self.pc_alpha = 0.6
        self.pc_boost_max_db = 8.0

        # Backward-compatible alias used by current step() getattr("pc_target_sinr_db", ...)
        self.pc_target_sinr_db = float(self.target_sinr_db)

        # Optional: keep the legacy μ array if any other code references it
        # (step() will not use it because mean_sinr_db exists)
        phy_cfg = getattr(PHYLayer, "cfg", None)
        base_mu = float(getattr(phy_cfg, "sinr_mu_db", 12.0)) if phy_cfg is not None else 12.0
        self.dev_sinr_mu_db = np.empty(self.n_devices, dtype=float)
        for i in range(self.n_devices):
            self.dev_sinr_mu_db[i] = float(base_mu) + float(self.rng.normal(0.0, 2.0))

        # queues
        self.queue_bits = np.zeros(self.n_devices, dtype=float)
        self.queue_age = np.zeros(self.n_devices, dtype=int)
        self._pkt_arrivals = [deque() for _ in range(self.n_devices)]

        # episode accumulators
        self.reset_metrics()
        self.t = 0
        self.ep_done = False
    # ------------------------------------------------------------
    def reset_metrics(self) -> None:
        self.bits_tx_ep = 0.0
        self.energy_j_ep = 0.0
        self.arrivals_this_ep = 0
        self.blocked_arrivals_ep = 0
        self.denies_this_ep = 0
        self.preemptions_ep = 0
        self.preempt_this_ep = 0   # alias
        self.scheduled_tx_ep = 0
        self.slots_used_ep = 0

        self.delay_samples_ms_ep: list[float] = []
        self.served_bits_per_device_ep = np.zeros(self.n_devices)
        self.served_bits_per_class_ep = np.zeros(self.n_classes)

        # selection diagnostics
        self.selected_dev_counts_ep = np.zeros(self.n_devices, dtype=np.int64)
    def _prepare_decision(self):
        """Generate current-slot traffic and select candidate before controller observes it.

        Fast fading and interruption are sampled only during step(action).
        This opt-in two-stage timing is shared by every v7 policy.
        """
        self._pending_arrivals, _ = self._arrivals_step()
        self._pending_device = self._select_device_idx()

    def _energy_observation(self):
        """Causal, pre-arrival information only; no current-step fading/outcome leakage.

        Device-specific nominal energy is based on configured TX power and per-bit
        energy. No current-slot channel realization or future arrivals are used.
        """
        original = np.concatenate([
            self.queue_bits / (self.bits_per_sample + 1e-9),
            self.queue_age / max(self.steps_per_ep, 1),
        ])
        if not getattr(self, 'energy_features_enabled', False):
            return original.astype(np.float32)
        nominal_w = 10.0 ** ((self.tx_power_dbm - 30.0) / 10.0)
        # Normalized nominal transmit cost, queue urgency, and energy already spent.
        nominal_j = (nominal_w + self.circuit_power_w) * self.slot_s + self.energy_per_bit * self.bits_per_sample
        cost = np.clip(nominal_j / 0.001, 0.0, 10.0)
        backlog = (self.queue_bits > 0).astype(float)
        used = float(self.energy_j_ep) / max(self.steps_per_ep * float(getattr(self, "energy_obs_budget_j", 0.000343)), 1e-12)
        remaining = 1.0 - float(self.t) / max(self.steps_per_ep, 1)
        features = np.concatenate([original, cost, backlog, [used, remaining]])
        if getattr(self, 'predecision_candidate_enabled', False):
            idx = int(self._pending_device)
            sinr_est = float(self.mean_sinr_db[idx] + self.shadow_db_ep[idx])
            interfer = float(self.interf_w_ep / max(self.interf_w_base, 1e-15))
            # Selected-device features are causally available before TX action.
            # SINR is a large-scale estimate; fast fading remains hidden.
            features = np.concatenate([features, [
                idx / max(self.n_devices - 1, 1),
                np.clip(self.queue_bits[idx] / max(self.bits_per_sample[idx] * 10., 1.), 0, 10),
                self.queue_age[idx] / max(self.steps_per_ep, 1),
                cost[idx],
                np.clip(sinr_est / 30., -2., 2.),
                np.clip(np.log(max(interfer, 1e-9)), -5., 5.),
                float(self.queue_bits[idx] > 0),
            ]])
        return features.astype(np.float32)

    # ------------------------------------------------------------
    def reset(self):
        self.queue_bits[:] = 0
        self.queue_age[:] = 0
        self._pkt_arrivals = [deque() for _ in range(self.n_devices)]
        self.battery_j = self.battery_cap_j.copy()

        self.reset_metrics()
        self.t = 0
        self._rr_ptr = 0
        self.ep_done = False

        # ------------------------------------------------------------
        # Episode-stable channel randomness (draw once per episode)
        # ------------------------------------------------------------
        # Shadowing: fixed per device within an episode (dB)
        # If you want stronger diversity, increase std from 6.0 -> 7.0/8.0
        shadow_std_db = float(getattr(self, "shadow_std_db", 6.0))
        self.shadow_db_ep = self.rng.normal(0.0, shadow_std_db, size=self.n_devices).astype(float)

        # Background interference: fixed scalar within an episode (W), mildly variable across episodes
        # Increase sigma_ln to widen EE/energy-penalty distribution.
        base = float(getattr(self, "interf_w_base", 1e-10))
        sigma_ln = float(getattr(self, "interf_sigma_ln", 0.6))  # 0.4 mild, 0.7 stronger
        self.interf_w_ep = float(base * np.exp(self.rng.normal(0.0, sigma_ln)))

        # ------------------------------------------------------------
        # observation: queue bits + ages
        # ------------------------------------------------------------
        if getattr(self, 'predecision_candidate_enabled', False):
            self._prepare_decision()
        obs = self._energy_observation()
        return obs
    # ------------------------------------------------------------
    def _select_device_idx(self) -> int:
        backlogged = np.where(self.queue_bits > 0)[0]
        if backlogged.size == 0:
            idx = int(self.t % self.n_devices)
            self.selected_dev_counts_ep[idx] += 1
            return idx

        ages = self.queue_age[backlogged].astype(float)
        bits = self.queue_bits[backlogged].astype(float)
        pr = np.array([self._prio_w[int(self.dev_class[i])] for i in backlogged], dtype=float)

        mode = str(getattr(self, "device_select_mode", "max_weight"))

        if mode == "round_robin":
            # True RR over all devices, but pick the first backlogged device encountered
            if not hasattr(self, "_rr_ptr"):
                self._rr_ptr = int(self.t % self.n_devices)

            start = int(self._rr_ptr) % self.n_devices
            chosen = None
            for k in range(self.n_devices):
                cand = (start + k) % self.n_devices
                if self.queue_bits[cand] > 0:
                    chosen = cand
                    self._rr_ptr = (cand + 1) % self.n_devices
                    break

            idx = int(chosen if chosen is not None else backlogged[0])

        elif mode == "max_age":
            score = ages + 0.01 * pr
            # small jitter avoids deterministic ties collapsing to lowest index
            score = score + 1e-6 * self.rng.random(score.shape[0])
            idx = int(backlogged[int(np.argmax(score))])

        else:  # "max_weight" (default)
            urgency = 0.75 * (ages / max(self.steps_per_ep, 1)) + 0.25 * np.log1p(bits)
            score = pr * urgency
            score = score + 1e-6 * self.rng.random(score.shape[0])
            idx = int(backlogged[int(np.argmax(score))])

        self.selected_dev_counts_ep[idx] += 1
        return idx
    def _arrivals_step(self) -> Tuple[int, int]:
        """
        Poisson arrivals with optional burstiness (Gamma multiplier) and per-device heterogeneity.

        Returns:
            arrivals_this_step, blocked_this_step
        """
        lam_dt = np.clip(self.sampling_hz * float(SLOT_DURATION_S), 0.0, 200.0)
        lam_dt = lam_dt * getattr(self, "arrival_mult", 1.0)

        bk = float(getattr(self, "burst_k", 0.0) or 0.0)
        if bk > 0.0:
            burst = float(self.rng.gamma(shape=bk, scale=1.0 / bk))  # mean=1
            lam_dt = lam_dt * burst

        cap = 3
        if bk > 0.0:
            cap = max(cap, 12)
        a_scale = float(getattr(self, "arrival_scale", 0.0) or 0.0)
        if a_scale > 0.0:
            cap = max(cap, int(np.ceil(3.0 * max(1.0, a_scale / 150.0))))
        lam_dt = np.clip(lam_dt, 0.0, float(cap))

        counts = self.rng.poisson(lam=lam_dt)
        arrivals_this_step = int(np.sum(counts))

        # Admit only complete packets that fit the bit-capacity budget.
        # Packet records store (arrival_slot, residual_bits). FIFO service may
        # span several slots or complete several packets in one slot.
        blocked_this_step = 0
        for i in np.where(counts > 0)[0]:
            packet_size = float(self.bits_per_sample[i])
            for _ in range(int(counts[i])):
                if float(self.queue_bits[i]) + packet_size <= self.queue_cap_bits + 1e-9:
                    self._pkt_arrivals[i].append([int(self.t), packet_size])
                    self.queue_bits[i] += packet_size
                else:
                    blocked_this_step += 1

        # Ages advance once at slot completion, not again during arrivals.
        # New packets are age zero at their first service opportunity.
        self.arrivals_this_ep += int(arrivals_this_step)
        self.blocked_arrivals_ep += int(blocked_this_step)
        return arrivals_this_step, blocked_this_step
    # ------------------------------------------------------------
    def step(self, action: int):
        if self.ep_done:
            obs = self._energy_observation()
            return obs, 0.0, True, {}

        if getattr(self, 'three_action_enabled', False) and action not in (0, 1, 2):
            raise ValueError('three-action environment requires action in {0,1,2}')
        # ------------------------------------------------------------
        # arrivals (Poisson + burstiness + heterogeneity)
        # ------------------------------------------------------------
        if getattr(self, 'predecision_candidate_enabled', False):
            arrivals_step = self._pending_arrivals
            dev = self._pending_device
        else:
            arrivals_step, _blocked_arrivals_step = self._arrivals_step()
            dev = self._select_device_idx()
        cls = int(self.dev_class[dev])

        backlog_bits = (self.queue_bits[dev] > 0)
        has_pkt = (len(self._pkt_arrivals[dev]) > 0)

        # Queue occupancy and the packet ledger must remain synchronized.
        if backlog_bits != has_pkt:
            raise AssertionError("packet ledger and bit queue are inconsistent")

        serviceable = bool(backlog_bits and has_pkt)

        delivered_bits = 0.0
        energy_j = 0.0
        delays = []
        preempted_step = 0
        outage_step = 0

        scheduled_step = int((action != 0) and serviceable)
        slots_used_step = scheduled_step

        # diagnostics (always defined, safe defaults)
        sinr_db_step = np.nan
        rate_mult_step = 0.0
        interrupt_prob_step = 0.0
        outage_prob_step = 0.0

        # ------------------------------------------------------------
        # transmit attempt
        # ------------------------------------------------------------
        # Fixed PHY innovations for every time slot, regardless of chosen action.
        fading_z = float(self.phy_rng.normal())
        interruption_u = float(self.phy_rng.random())
        if scheduled_step:
            pkt_bits_nom = float(self.bits_per_sample[dev])

            # ---------- build instantaneous SINR (policy-coupled via device selection) ----------
            # Base (device) mean SINR
            if hasattr(self, "mean_sinr_db"):
                base_mu = float(self.mean_sinr_db[dev])
            elif hasattr(self, "dev_sinr_mu_db"):
                base_mu = float(self.dev_sinr_mu_db[dev])
            elif hasattr(self, "base_sinr_db"):
                base_mu = float(self.base_sinr_db[dev])
            else:
                base_mu = 12.0

            # Episode-stable shadowing for this device (dB)
            sh = 0.0
            if hasattr(self, "shadow_db_ep"):
                sh = float(self.shadow_db_ep[dev])
            elif hasattr(self, "shadow_db"):
                sh = float(self.shadow_db[dev])

            # Episode-stable interference penalty (convert W scalar to a small dB penalty)
            interf_pen_db = 0.0
            try:
                base_i = float(getattr(self, "interf_w_base", 1e-10))
                ep_i = float(getattr(self, "interf_w_ep", base_i))
                ratio = ep_i / max(base_i, 1e-15)
                interf_pen_db = 2.0 * float(np.log10(max(ratio, 1e-12)))
                interf_pen_db = float(np.clip(interf_pen_db, -3.0, 6.0))
            except Exception:
                interf_pen_db = 0.0

            # Fast fading around the large-scale mean
            sigma_db = float(getattr(self, "sinr_sigma_db", 3.0))
            # Lower TX power decreases link SINR by the same dB amount.
            power_offset_db = -float(getattr(self, 'low_power_backoff_db', 6.0)) if (getattr(self, 'three_action_enabled', False) and action == 2) else 0.0
            sinr_db = float(base_mu + sh - interf_pen_db + sigma_db * fading_z + power_offset_db)

            rate_mult = PHYLayer.rate_multiplier_from_sinr_db(float(sinr_db))

            sinr_db_step = float(sinr_db)
            rate_mult_step = float(rate_mult)

            # ---------- mild power control (affects TX power/energy, not SINR directly) ----------
            tx_boost_db = 0.0
            if bool(getattr(self, "enable_power_control", False)):
                pc_alpha = float(getattr(self, "pc_alpha", 0.6))
                pc_target = float(getattr(self, "target_sinr_db", getattr(self, "pc_target_sinr_db", 8.0)))
                pc_boost_max = float(getattr(self, "pc_boost_max_db", 8.0))
                tx_boost_db = PHYLayer.power_control_boost_db(
                    sinr_db_step, pc_target, pc_alpha, pc_boost_max
                )

            # Low-power backoff is applied to radiated power, and the model
            # records the corresponding SINR reduction. Freeze TX boost in
            # low-power mode to prevent the controller cancelling its own backoff.
            if getattr(self, 'three_action_enabled', False) and action == 2:
                tx_boost_db = 0.0
            tx_p_dbm_eff = float(self.tx_power_dbm[dev]) + float(tx_boost_db) + power_offset_db

            # ---------- interruption model ----------
            p_interrupt = 0.01 + 0.03 * float(self.drag_penalty[dev])
            p_interrupt += 0.05 * float(np.clip((8.0 - sinr_db_step) / 12.0, 0.0, 1.0))
            p_interrupt = float(np.clip(p_interrupt, 0.0, 0.55))
            interrupt_prob_step = float(p_interrupt)

            interrupted = (interruption_u < p_interrupt)

            # ---------- effective bits (link adaptation) ----------
            pkt_bits_eff = pkt_bits_nom * float(np.clip(rate_mult_step, 0.05, 1.45))

            # ---------- outage model (optional, safe if required_sinr_db_for_bits absent) ----------
            enable_outage = bool(getattr(self, "enable_outage", False))
            in_outage = False
            if enable_outage and hasattr(PHYLayer, "required_sinr_db_for_bits"):
                req_sinr_db = PHYLayer.required_sinr_db_for_bits(int(max(0.0, pkt_bits_eff)), bw_hz=1e6)
                outage_margin_db = float(getattr(self, "outage_margin_db", 1.5))
                in_outage = bool(sinr_db_step < (float(req_sinr_db) + float(outage_margin_db)))
            outage_prob_step = float(1.0 if in_outage else 0.0)

            # ---------- energy accounting (TX + circuit) ----------
            circuit_w = float(getattr(self, "circuit_power_w", getattr(self, "tx_circuit_w", 0.0)))
            e_tx_full = PHYLayer.energy_joules(
                tx_p_dbm_eff,
                float(SLOT_DURATION_S),
                circuit_power_w=float(circuit_w),
            )

            if interrupted:
                preempted_step = 1
                self.preemptions_ep += 1

                # context-coupled waste fraction (stronger effect size than fixed 0.9)
                backlog_norm = float(np.clip(self.queue_bits[dev] / max(1.0, 3.0 * pkt_bits_nom), 0.0, 1.0))
                age_norm = float(np.clip(self.queue_age[dev] / max(1.0, self.steps_per_ep), 0.0, 1.0))
                sinr_bad = float(np.clip((10.0 - sinr_db_step) / 20.0, 0.0, 1.0))

                waste_frac = 0.65 + 0.20 * backlog_norm + 0.10 * age_norm + 0.05 * sinr_bad
                waste_frac = float(np.clip(waste_frac, 0.60, 0.98))

                delivered_bits = 0.0
                energy_j = float(e_tx_full) * float(waste_frac)
                # packet stays in queue

            elif in_outage:
                outage_step = 1
                self.outages_ep = int(getattr(self, "outages_ep", 0)) + 1

                waste_frac = float(getattr(self, "outage_waste_frac", 0.70))
                delivered_bits = 0.0
                energy_j = float(e_tx_full) * float(np.clip(waste_frac, 0.0, 1.0))
                # packet stays in queue

            else:
                # Serve available bits from the FIFO packet ledger, retaining
                # the unserved portion of any partially transmitted packet.
                available = float(min(pkt_bits_eff, float(self.queue_bits[dev])))
                outstanding = available
                while outstanding > 1e-9:
                    arrival_slot, residual = self._pkt_arrivals[dev][0]
                    consumed = min(outstanding, residual)
                    outstanding -= consumed
                    residual -= consumed
                    if residual <= 1e-9:
                        self._pkt_arrivals[dev].popleft()
                        delays.append(float((self.t - arrival_slot + 1) * SLOT_DURATION_S * 1e3))
                    else:
                        self._pkt_arrivals[dev][0][1] = residual
                delivered_bits = available
                self.queue_bits[dev] = max(0.0, float(self.queue_bits[dev]) - delivered_bits)

                # Energy: active-slot energy + per-bit processing/radio energy
                energy_j = float(e_tx_full) + delivered_bits * float(self.energy_per_bit[dev])

                self.served_bits_per_device_ep[dev] += delivered_bits
                self.served_bits_per_class_ep[cls] += delivered_bits

        else:
            # deny with serviceable backlog: charge sensing/decision energy (policy-dependent)
            if (action == 0) and serviceable:
                self.denies_this_ep += 1
                self.blocked_arrivals_ep += 1
                energy_j = float(getattr(self, "sense_power_w", 0.0)) * float(SLOT_DURATION_S)

        # ------------------------------------------------------------
        # accounting
        # ------------------------------------------------------------
        self.bits_tx_ep += delivered_bits
        self.energy_j_ep += energy_j
        self.delay_samples_ms_ep.extend(delays)

        self.scheduled_tx_ep += scheduled_step
        self.slots_used_ep += slots_used_step

        self.queue_age[self.queue_bits > 0] += 1
        self.queue_age[self.queue_bits <= 0] = 0
        # Detect violations at source instead of silently modifying queues.
        for i in range(self.n_devices):
            ledger_bits = sum(packet[1] for packet in self._pkt_arrivals[i])
            if abs(ledger_bits - float(self.queue_bits[i])) > 1e-6:
                raise AssertionError(f"queue ledger mismatch for device {i}: {ledger_bits} vs {self.queue_bits[i]}")
            if self.queue_bits[i] > self.queue_cap_bits + 1e-6:
                raise AssertionError("queue overflow")

        self.t += 1
        self.ep_done = self.t >= self.steps_per_ep
        if getattr(self, 'predecision_candidate_enabled', False) and not self.ep_done:
            self._prepare_decision()
        obs = self._energy_observation()

        blocked_step = int((action == 0) and serviceable)

        # ------------------------------------------------------------
        # info (created only here, after all variables exist)
        # ------------------------------------------------------------
        info = {
            "arrivals": int(arrivals_step),
            "blocked": int(blocked_step),
            "preempted": int(preempted_step),
            "outage": int(outage_step),
            "scheduled": int(scheduled_step),
            "slots_used": int(slots_used_step),
            "delivered_bits": float(delivered_bits),
            "energy_j": float(energy_j),
            "interrupt_prob": float(interrupt_prob_step),
            "outage_prob": float(outage_prob_step),
            "p_interrupt": float(interrupt_prob_step),  # optional alias for newer code paths
            "sinr_db": float(sinr_db_step) if np.isfinite(sinr_db_step) else np.nan,
            "rate_mult": float(rate_mult_step),
            "tx_mode": 'low' if action == 2 else ('normal' if action == 1 else 'deny'),
        }

        if delays:
            info["delay_ms"] = float(delays[-1])

        served_bits_per_class_step = np.zeros(self.n_classes, dtype=float)
        served_bits_per_device_step = np.zeros(self.n_devices, dtype=float)
        if delivered_bits > 0:
            served_bits_per_class_step[cls] = float(delivered_bits)
            served_bits_per_device_step[int(dev)] = float(delivered_bits)

        info["served_bits_per_class"] = served_bits_per_class_step
        info["served_bits_per_device"] = served_bits_per_device_step

        return obs, float(delivered_bits), self.ep_done, info
    # ------------------------------------------------------------
    def _rollup_episode_metrics(self) -> Dict[str, float]:
        T = max(self.steps_per_ep * SLOT_DURATION_S, 1e-9)

        thr_mbps = float((self.bits_tx_ep / T) / 1e6)
        avg_delay_ms = float(np.mean(self.delay_samples_ms_ep)) if self.delay_samples_ms_ep else 0.0
        energy_eff_bpj = float(self.bits_tx_ep / self.energy_j_ep) if self.energy_j_ep > 0 else 0.0

        blocking_prob = float(self.denies_this_ep / max(self.arrivals_this_ep, 1))
        interrupt_prob = float(self.preemptions_ep / max(self.scheduled_tx_ep, 1))

        util = float(self.bits_tx_ep / (self.n_channels * T * CHANNEL_RATE_BPS))

        # selection diagnostics
        c = self.selected_dev_counts_ep.astype(float)
        tot = float(c.sum())
        if tot > 0:
            p = c / tot
            pp = p[p > 0]
            sel_entropy = float(-np.sum(pp * np.log(pp + 1e-12)))
            sel_top = float(np.max(p))
            sel_active = int(np.sum(c > 0))
        else:
            sel_entropy = 0.0
            sel_top = 0.0
            sel_active = 0

        def _jain(x: np.ndarray) -> float:
            x = np.asarray(x, dtype=float).ravel()
            x = x[np.isfinite(x)]
            if x.size == 0:
                return 0.0
            s1 = float(np.sum(x))
            s2 = float(np.sum(x * x))
            if s2 <= 0.0:
                return 0.0
            return float((s1 * s1) / (x.size * s2))

        jain_classes = _jain(self.served_bits_per_class_ep)
        jain_devices = _jain(self.served_bits_per_device_ep)

        return {
            "throughput_mbps": thr_mbps,
            "avg_delay_ms": avg_delay_ms,
            "energy_eff_bpj": energy_eff_bpj,
            "blocking_prob": blocking_prob,
            "interrupt_prob": interrupt_prob,
            "utilization": util,
            "sel_entropy": sel_entropy,
            "sel_top_share": sel_top,
            "sel_active_devices": float(sel_active),
            "jain_fairness_classes": jain_classes,
            "jain_fairness_devices": jain_devices,
        }

# ============================================================
# Multi-agent wrapper (ONLY env exposed to training)
# ============================================================

class MultiAgentSpectrumEnv:
    """
    Final multi-agent environment.
    This is the ONLY env that sim_runner / RL code should import.
    """

    def __init__(self, *, n_agents: int, base_env_kwargs: dict, reward_mode: str = "local"):

        self.n_agents = int(n_agents)
        self.reward_mode = reward_mode

        # build agents FIRST
        self.agents: Dict[int, SpectrumEnv] = {}
        for aid in range(self.n_agents):
            k = dict(base_env_kwargs)
            k["seed"] = int(base_env_kwargs.get("seed", 0)) + aid
            self.agents[aid] = SpectrumEnv(**k)

        # infer shared dimensions from a reference agent
        ref = next(iter(self.agents.values()))
        self.n_devices = ref.n_devices
        self.n_channels = ref.n_channels
        self.steps_per_ep = ref.steps_per_ep
        self.slot_s = ref.slot_s

        self.n_actions = 2
        self.action_dim = 2

        self.ep_done = False
        self.episode_metrics: Dict[str, float] | None = None

    # ------------------------------------------------------------


    def reset(self):
        self.ep_done = False
        self.episode_metrics = None

        obs = {}
        for aid, env in self.agents.items():
            o = env.reset()  # already returns the correct numeric observation
            obs[aid] = np.asarray(o, dtype=np.float32).reshape(-1)

        return obs

    # ------------------------------------------------------------


    def step(self, actions: Dict[int, int]):
        obs, rewards, dones = {}, {}, {}
        infos = {}

        for aid, env in self.agents.items():
            o, r, d, info = env.step(actions.get(aid, 0))

            if o is None:
                # Defensive fallback: should not happen after SpectrumEnv.step fix,
                # but keep it to prevent dtype=object propagation.
                o = env._energy_observation()

            obs[aid] = np.asarray(o, dtype=np.float32).reshape(-1)
            rewards[aid] = float(r)
            dones[aid] = bool(d)
            infos[aid] = info

        self.ep_done = all(dones.values())

        if self.reward_mode == "global":
            rmean = float(np.mean(list(rewards.values()))) if rewards else 0.0
            rewards = {aid: rmean for aid in rewards}

        return obs, rewards, self.ep_done, infos



    def encode_state(self, obs_dict):
        """
        Flatten global state across agents.
        Used only by tabular methods.
        """
        if not isinstance(obs_dict, dict):
            return 0

        states = []
        for aid in sorted(obs_dict.keys()):
            o = obs_dict[aid]
            if o is None:
                continue
            arr = np.asarray(o, dtype=np.float32).reshape(-1)
            states.append(arr)

        if not states:
            return 0

        return tuple(np.concatenate(states).tolist())
