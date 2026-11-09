# sim_runner.py (FINAL, multi-agent only)
# ------------------------------------------------------------
# Runs all schemes under MultiAgentSpectrumEnv, sweeps device-selection modes,
# aggregates across seeds, writes per-scheme per-episode CSVs + per-mode summary CSV,
# and (optionally) produces plots if your utils.py exposes plotting helpers.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# ===== Project imports =====
from env import MultiAgentSpectrumEnv  # multi-agent only

# PHY constants (used for utilization sanity where needed)
try:
    from phy_layer import SLOT_DURATION_S, CHANNEL_RATE_BPS, PHYLayer
except Exception:
    SLOT_DURATION_S = 0.001
    CHANNEL_RATE_BPS = 1e6

# --- Final scheme trainers (multi-agent only) ---
from rl_tabular import (
    rl_qlearning_multiagent_train,
    rl_doubleq_multiagent_train,
)

from rl_dqn import (
    rl_dqn_multiagent_train,
    rl_dueling_dqn_multiagent_train,
    DQNCfg,
    DuelingCfg,
)

from rl_policy_gradient import (
    rl_policy_gradient_train,
    PolicyGradientConfig,
)


# Optional utilities module (plots, convergence, csv helpers)
try:
    import utils as U
except Exception:
    U = None

OUTDIR = Path("./outputs")
DEFAULT_MODE_SWEEP = ("round_robin", "max_age", "max_weight")

# Keep scheme order stable (paper-friendly)
SCHEME_NAMES = ["Q-Learning", "DoubleQ", "DQN", "PPO", "Proposed"]


# ============================================================
# Small local helpers (do not depend on utils.py)
# ============================================================

def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")
def _safe_mean(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    return float(finite.mean())
def _episodes_to_convergence_local(
    rewards: List[float],
    *,
    frac: float = 0.95,
    window: int = 20,
    band: float = 0.05,
    streak: int = 5,
) -> int:
    """
    Fallback if utils.episodes_to_convergence is unavailable.
    Simple: convergence when moving-average stays within +/- band of target
    for `streak` windows.
    """
    r = np.asarray(rewards, float)
    if r.size < max(window, 5):
        return int(r.size)

    # target = frac of max moving-average
    ma = np.convolve(r, np.ones(window) / window, mode="valid")
    tgt = frac * float(np.nanmax(ma)) if np.isfinite(np.nanmax(ma)) else float(np.nanmax(r))
    if not np.isfinite(tgt):
        return int(r.size)

    lo = (1.0 - band) * tgt
    hi = (1.0 + band) * tgt

    hit = 0
    for i, v in enumerate(ma, start=window):
        if lo <= v <= hi:
            hit += 1
            if hit >= streak:
                return int(i)
        else:
            hit = 0
    return int(r.size)
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)
def _save_metrics_csv_fallback(path: str, metrics: List[Dict[str, Any]]) -> None:
    """
    Only used if utils.save_metrics_csv is missing.
    Writes union of keys in stable order (approx).
    """
    if not metrics:
        return
    keys = []
    seen = set()
    preferred = [
        "throughput_mbps",
        "avg_delay_ms",
        "energy_eff_bpj",
        "jain_fairness",
        "jain_fairness_devices",
        "blocking_prob",
        "interrupt_prob",
        "utilization",
        "sel_entropy",
        "sel_top_share",
        "sel_active_devices",
        "bits_tx_ep",
        "energy_j",
        "total_arrivals_ep",
        "blocked_arrivals_ep",
        "scheduled_tx_ep",
        "preemptions_ep",
        "slots_used_ep",
        "qmax",
    ]
    for k in preferred:
        if any(k in m for m in metrics):
            keys.append(k)
            seen.add(k)
    for m in metrics:
        for k in m.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)

    outp = Path(path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for m in metrics:
            w.writerow({k: m.get(k, "") for k in keys})
def _save_metrics_csv(path: str, metrics: List[Dict[str, Any]]) -> None:
    if U is not None and hasattr(U, "save_metrics_csv"):
        U.save_metrics_csv(path, metrics)
    else:
        _save_metrics_csv_fallback(path, metrics)
def _episodes_to_convergence(rewards: List[float], window: int) -> int:
    if U is not None and hasattr(U, "episodes_to_convergence"):
        return int(U.episodes_to_convergence(rewards, frac=0.95, window=window, band=0.05, streak=5))
    return int(_episodes_to_convergence_local(rewards, window=window))

# ============================================================
# Environment factory (multi-agent only)
# ============================================================

def _apply_phy_overrides(env, *, sinr_sigma_db=None, device_jitter_cv=None):
    """
    Apply PHY/heterogeneity knobs after env construction so we do not require
    SpectrumEnv.__init__(...) to accept these kwargs.

    Works for MultiAgentSpectrumEnv (env.agents dict) and single envs.
    """
    targets = []
    if hasattr(env, "agents") and isinstance(getattr(env, "agents"), dict) and env.agents:
        targets.extend(list(env.agents.values()))
    else:
        targets.append(env)

    for e in targets:
        # SINR dispersion (shadowing / fading std in dB)
        if sinr_sigma_db is not None:
            try:
                if hasattr(e, "sinr_sigma_db"):
                    e.sinr_sigma_db = float(sinr_sigma_db)
                # common nested names
                for attr in ("phy", "phy_layer", "phy_obj"):
                    if hasattr(e, attr):
                        ph = getattr(e, attr)
                        if hasattr(ph, "sinr_sigma_db"):
                            ph.sinr_sigma_db = float(sinr_sigma_db)
            except Exception:
                pass

        # Per-device energy/tx jitter CV (attribute name may vary across versions)
        if device_jitter_cv is not None:
            try:
                for nm in ("device_jitter_cv", "energy_jitter_cv", "tx_power_jitter_cv"):
                    if hasattr(e, nm):
                        setattr(e, nm, float(device_jitter_cv))
                for attr in ("phy", "phy_layer", "phy_obj"):
                    if hasattr(e, attr):
                        ph = getattr(e, attr)
                        for nm in ("device_jitter_cv", "energy_jitter_cv", "tx_power_jitter_cv"):
                            if hasattr(ph, nm):
                                setattr(ph, nm, float(device_jitter_cv))
            except Exception:
                pass

    return env


def make_env(
    *,
    steps_per_ep: int,
    n_devices: int,
    seed: int,
    arrival_scale: float,
    class_mix: Tuple[float, float, float],
    burst_k: float | None,
    device_skew: float,
    bits_scale: float,
    delay_penalty_w: float,
    energy_penalty_w: float,
    prio_w: Tuple[float, float, float],
    device_select_mode: str,
    n_agents: int,
    n_channels: int = 1,
    sinr_sigma_db: float = 3.5,
    device_jitter_cv: float = 0.0,
) -> MultiAgentSpectrumEnv:
    """
    Multi-agent only. Each agent gets its own SpectrumEnv internally.

    IMPORTANT:
    - SpectrumEnv.__init__ DOES NOT accept sinr_sigma_db. That knob belongs to the PHY layer.
      Therefore we apply sinr_sigma_db via PHYLayer.set_cfg(...) and do NOT pass it inside
      base_env_kwargs.
    - SpectrumEnv.__init__ DOES accept device_jitter_cv, so we pass it through base_env_kwargs.
    """
    # normalize class_mix
    cm = np.asarray(class_mix, float).reshape(-1)
    if cm.size != 3:
        raise ValueError(f"class_mix must have 3 elements, got {class_mix}")
    s = float(cm.sum())
    class_mix_n = tuple((cm / s).tolist()) if s > 0 else (1 / 3, 1 / 3, 1 / 3)

    # robust burst_k
    if burst_k is None:
        burst_k_val = None
    else:
        if isinstance(burst_k, str) and burst_k.strip().lower() in {"none", "null", ""}:
            burst_k_val = None
        else:
            burst_k_val = float(burst_k)

    # Apply PHY configuration (global/static)
    try:
        PHYLayer.set_cfg(sinr_sigma_db=float(sinr_sigma_db))
    except Exception:
        pass

    base_env_kwargs = dict(
        steps_per_ep=int(steps_per_ep),
        n_devices=int(n_devices),
        n_channels=int(n_channels),
        seed=int(seed),
        arrival_scale=float(arrival_scale),
        class_mix=tuple(class_mix_n),
        burst_k=burst_k_val,
        device_skew=float(device_skew),
        bits_scale=float(bits_scale),
        delay_penalty_w=float(delay_penalty_w),
        energy_penalty_w=float(energy_penalty_w),
        prio_weights=tuple(prio_w),
        device_select_mode=str(device_select_mode),
        device_jitter_cv=float(device_jitter_cv),
    )

    return MultiAgentSpectrumEnv(
        n_agents=int(n_agents),
        base_env_kwargs=base_env_kwargs,
        reward_mode="local",
    )

# ============================================================
# Aggregation across seeds (per mode)
# ============================================================

def aggregate_over_seeds(all_runs: List[List[Tuple[str, Dict[str, Any]]]]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    all_runs: list over seeds, each element = results list [(scheme, metrics_dict), ...]
    returns list of (scheme, agg_metrics_dict) with:
      - episode_metrics_std: list of per-episode dict with mean/std merged (mean in values; std kept in *_std keys if desired)
      - episode_metrics_mean: dict of mean over episodes
    """
    by_scheme: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for run in all_runs:
        for name, m in run:
            if isinstance(m, dict):
                by_scheme[name].append(m)
            else:
                by_scheme[name].append({})

    agg: List[Tuple[str, Dict[str, Any]]] = []

    for name in SCHEME_NAMES:
        ms = by_scheme.get(name, [])
        if not ms:
            agg.append((name, {"episode_metrics_std": []}))
            continue

        # collect per-episode standardized metrics
        eps_lists: List[List[Dict[str, Any]]] = []
        for m in ms:
            eps = m.get("episode_metrics_std", []) or []
            eps_lists.append(eps if isinstance(eps, list) else list(eps))

        max_len = max((len(eps) for eps in eps_lists), default=0)
        if max_len <= 0:
            agg.append((name, {"episode_metrics_std": []}))
            continue

        # union of keys across all episodes / seeds
        keyset = set()
        for eps in eps_lists:
            for row in eps:
                if isinstance(row, dict):
                    keyset.update(row.keys())
        keys = sorted(keyset)

        # per-episode mean (and optional std)
        ep_mean_rows: List[Dict[str, Any]] = []
        for t in range(max_len):
            row_mean: Dict[str, Any] = {}
            for k in keys:
                vals = []
                for eps in eps_lists:
                    if t < len(eps):
                        v = eps[t].get(k, np.nan) if isinstance(eps[t], dict) else np.nan
                        try:
                            vals.append(float(v))
                        except Exception:
                            vals.append(np.nan)
                a = np.asarray(vals, float)
                row_mean[k] = _safe_mean(a)
            ep_mean_rows.append(row_mean)

        # overall mean over episodes (for convenience)
        overall: Dict[str, float] = {}
        for k in keys:
            a = np.asarray([float(r.get(k, np.nan)) for r in ep_mean_rows], float)
            overall[k] = _safe_mean(a)

        agg.append((name, {"episode_metrics_std": ep_mean_rows, "episode_metrics_mean": overall}))

    # keep any CMDP variants if present
    for name in list(by_scheme.keys()):
        if name in SCHEME_NAMES:
            continue
        # keep stable ordering for CMDP suffixes
        if name.endswith("_CMDP"):
            ms = by_scheme[name]
            eps_lists = []
            for m in ms:
                eps = m.get("episode_metrics_std", []) or []
                eps_lists.append(eps if isinstance(eps, list) else list(eps))
            max_len = max((len(eps) for eps in eps_lists), default=0)
            if max_len <= 0:
                agg.append((name, {"episode_metrics_std": []}))
                continue
            keyset = set()
            for eps in eps_lists:
                for row in eps:
                    if isinstance(row, dict):
                        keyset.update(row.keys())
            keys = sorted(keyset)
            ep_mean_rows = []
            for t in range(max_len):
                row_mean = {}
                for k in keys:
                    vals = []
                    for eps in eps_lists:
                        if t < len(eps):
                            v = eps[t].get(k, np.nan) if isinstance(eps[t], dict) else np.nan
                            try:
                                vals.append(float(v))
                            except Exception:
                                vals.append(np.nan)
                    row_mean[k] = _safe_mean(np.asarray(vals, float))
                ep_mean_rows.append(row_mean)
            overall = {k: _safe_mean(np.asarray([r.get(k, np.nan) for r in ep_mean_rows], float)) for k in keys}
            agg.append((name, {"episode_metrics_std": ep_mean_rows, "episode_metrics_mean": overall}))

    return agg


# ============================================================
# Per-mode summary CSV writer (includes selection diagnostics)
# ============================================================

def write_summary_csv(all_res: List[Tuple[str, Dict[str, Any]]], ts2: str, outdir: Path) -> None:
    out = outdir / f"summary_PHY_{ts2}.csv"
    fields = [
        "scheme",
        "avg_throughput_mbps",
        "avg_delay_ms",
        "avg_energy_eff_bits_per_j",
        "avg_fairness",
        "mean_blocking_prob",
        "mean_interrupt_prob",
        "mean_utilization",
        "mean_sel_entropy",
        "mean_sel_top_share",
        "mean_sel_active_devices",
    ]

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for name, m in all_res:
            eps = m.get("episode_metrics_std", []) or []
            if not eps:
                row_out = {k: 0.0 for k in fields}
                row_out["scheme"] = name
                w.writerow(row_out)
                continue

            thr = np.array([r.get("throughput_mbps", np.nan) for r in eps], float)
            dly = np.array([r.get("avg_delay_ms", np.nan) for r in eps], float)
            ee = np.array([r.get("energy_eff_bpj", np.nan) for r in eps], float)
            far = np.array([r.get("jain_fairness_devices", r.get("jain_fairness", np.nan)) for r in eps], float)
            bp = np.array([r.get("blocking_prob", np.nan) for r in eps], float)
            ip = np.array([r.get("interrupt_prob", np.nan) for r in eps], float)
            utl = np.array([r.get("utilization", np.nan) for r in eps], float)

            sel_ent = np.array([r.get("sel_entropy", np.nan) for r in eps], float)
            sel_top = np.array([r.get("sel_top_share", np.nan) for r in eps], float)
            sel_act = np.array([r.get("sel_active_devices", np.nan) for r in eps], float)

            w.writerow(
                dict(
                    scheme=name,
                    avg_throughput_mbps=_safe_mean(thr),
                    avg_delay_ms=_safe_mean(dly),
                    avg_energy_eff_bits_per_j=_safe_mean(ee),
                    avg_fairness=_safe_mean(far),
                    mean_blocking_prob=_safe_mean(bp),
                    mean_interrupt_prob=_safe_mean(ip),
                    mean_utilization=_safe_mean(utl),
                    mean_sel_entropy=_safe_mean(sel_ent),
                    mean_sel_top_share=_safe_mean(sel_top),
                    mean_sel_active_devices=_safe_mean(sel_act),
                )
            )

# ============================================================
# Core: run all schemes for one seed (multi-agent only)
# ============================================================

def run_for_seed(
    args: argparse.Namespace,
    seed: int,
    ts: str,
    dev: torch.device,
    outdir: Path,
) -> Tuple[List[Tuple[str, Dict[str, Any]]], Dict[str, float], Dict[str, int], int]:
    """
    Build a fresh multi-agent env per scheme (no cross-scheme contamination),
    run all schemes, write per-scheme CSVs, and return aggregated metrics.
    """

    train_time_s: Dict[str, float] = {}
    conv_ep: Dict[str, int] = {}
    results: List[Tuple[str, Dict[str, Any]]] = []

    # --------------------------------------------------------
    # Env builder (ALWAYS multi-agent)
    # --------------------------------------------------------
    def _build_env() -> MultiAgentSpectrumEnv:
        # Allow n_channels to be configured later without touching this function again.
        n_ch = int(getattr(args, "n_channels", 1))

        return make_env(
            steps_per_ep=int(args.steps),
            n_devices=int(args.n_devices),
            n_channels=int(n_ch),
            seed=int(seed),
            arrival_scale=float(args.arrival_scale),
            class_mix=tuple(args.class_mix),
            burst_k=args.burst_k,
            device_skew=float(args.device_skew),
            bits_scale=float(args.bits_scale),
            delay_penalty_w=float(args.delay_penalty_w),
            energy_penalty_w=float(args.energy_penalty_w),
            prio_w=tuple(args.prio_w),
            device_select_mode=str(args.device_select_mode),
            n_agents=int(args.n_agents),

            # SAFE: make_env() applies these post-construction (no SpectrumEnv kwargs)
            sinr_sigma_db=float(getattr(args, "sinr_sigma_db", 3.5)),
            device_jitter_cv=float(getattr(args, "device_jitter_cv", 0.0)),
        )

    # Reference env only to infer static parameters
    env_ref = _build_env()
    n_channels = int(getattr(env_ref, "n_channels", getattr(args, "n_channels", 1)))

    # --------------------------------------------------------
    # Scheme registry (FINAL, unified)
    # --------------------------------------------------------
    scheme_fns = {
        "Q-Learning": rl_qlearning_multiagent_train,
        "DoubleQ": rl_doubleq_multiagent_train,
        "DQN": rl_dqn_multiagent_train,
        "DuelingDQN": rl_dueling_dqn_multiagent_train,

        # Policy-gradient family (single implementation)
        "ActorCritic": lambda env, **kw: rl_policy_gradient_train(
            env,
            cfg=PolicyGradientConfig(algo="a2c"),
            **kw,
        ),
        "PPO": lambda env, **kw: rl_policy_gradient_train(
            env,
            cfg=PolicyGradientConfig(algo="ppo"),
            **kw,
        ),

        # Paper label only (same engine as PPO for now)
        "Proposed": lambda env, **kw: rl_policy_gradient_train(
            env,
            cfg=PolicyGradientConfig(
                algo="ppo",
                ent_coef=0.008,
                clip_eps=0.20,
                lam=0.95,
            ),
            **kw,
        ),
    }

    # --------------------------------------------------------
    # Run schemes
    # --------------------------------------------------------
    for name in SCHEME_NAMES:
        env = _build_env()
        print(f"Running {name} (multi-agent)...")

        t0 = time.perf_counter()
        m: Dict[str, Any] = {}

        try:
            fn = scheme_fns[name]

            if name in {"Q-Learning", "DoubleQ"}:
                # tabular: no torch device
                m = fn(
                    env,
                    episodes=int(args.episodes),
                    steps_per_ep=int(args.steps),
                )
            else:
                # neural methods
                m = fn(
                    env,
                    episodes=int(args.episodes),
                    steps_per_ep=int(args.steps),
                    device=dev,
                )

            if not isinstance(m, dict):
                m = {}

        except Exception as e:
            msg = f"{name} failed: {e.__class__.__name__}: {e}"
            print(f"[WARN] {msg}")

            # Persist the error so Proposed-missing runs are diagnosable.
            try:
                err_path = outdir / f"ERROR_{name}_{ts}_seed{seed}.txt"
                err_path.write_text(msg + "\n")
            except Exception:
                pass

            # If you want to HARD-FAIL when Proposed fails (recommended for final run),
            # uncomment the next two lines.
            # if name == "Proposed":
            #     raise

            m = {}

        finally:
            train_time_s[name] = time.perf_counter() - t0

        # ----------------------------------------------------
        # Integrity + convergence + CSV
        # ----------------------------------------------------
        eps = m.get("episode_metrics_std", []) if isinstance(m, dict) else []
        eps = eps or []

        if isinstance(eps, list) and eps:
            tot_arr = sum(int(r.get("arrivals", 0) or 0) for r in eps)
            tot_sched = sum(int(r.get("scheduled", 0) or 0) for r in eps)
            tot_slots = sum(int(r.get("slots_used", 0) or 0) for r in eps)
            tot_bits = sum(float(r.get("bits", 0.0) or 0.0) for r in eps)

            print(
                f"[INTEGRITY] {name}: "
                f"arrivals={tot_arr} scheduled={tot_sched} "
                f"slots={tot_slots} bits={tot_bits:.1f}"
            )

            # convergence (reward-based)
            rewards = m.get("rewards_per_episode", []) if isinstance(m, dict) else []
            rewards = rewards or []
            if isinstance(rewards, list) and rewards:
                conv_ep[name] = int(_episodes_to_convergence(rewards, window=int(args.window)))

            # per-scheme CSV
            csv_path = outdir / f"{name}_metrics_PHY_{ts}_seed{seed}.csv"
            _save_metrics_csv(csv_path, eps)
        else:
            # Do not write a fake CSV here. If the scheme failed, better to keep it absent
            # and rely on the ERROR_* file for debugging, otherwise you'll get misleading
            # zero/empty artifacts in summaries.
            pass

        results.append((name, m if isinstance(m, dict) else {}))

    return results, train_time_s, conv_ep, n_channels

# ============================================================
# Main
# ============================================================

def main() -> None:
    global args

    p = argparse.ArgumentParser()

    # Keep argparse (so it still works from CMD), but defaults are now paper-facing.
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--n_devices", type=int, default=12)

    # multi-seed
    p.add_argument("--seeds", type=str, default="1,2,3,4,5")

    # Paper-facing stress defaults
    p.add_argument("--arrival_scale", type=float, default=1600.0)
    p.add_argument("--bits_scale", type=float, default=2.0)
    p.add_argument("--delay_penalty_w", type=float, default=0.012)
    p.add_argument("--energy_penalty_w", type=float, default=0.00040)

    # PHY / heterogeneity knobs (consumed by make_env -> env/phy post-construction config)
    p.add_argument("--sinr_sigma_db", type=float, default=4.5)
    p.add_argument("--device_jitter_cv", type=float, default=0.25)

    # Optional: channels if you ever want >1 later; safe default is 1
    p.add_argument("--n_channels", type=int, default=1)

    p.add_argument("--prio_w", type=float, nargs=3, default=(1.3, 1.0, 0.9))
    p.add_argument("--class_mix", type=float, nargs=3, default=(0.55, 0.30, 0.15))
    p.add_argument("--burst_k", type=float, default=4.0)
    p.add_argument("--device_skew", type=float, default=1.35)

    # Keep a single-mode arg for debugging, but paper run sweeps all 3 modes regardless.
    p.add_argument(
        "--device_select_mode",
        type=str,
        default="max_weight",
        choices=["round_robin", "max_age", "max_weight"],
    )

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # multi-agent only
    p.add_argument("--n_agents", type=int, default=5)

    # plotting
    p.add_argument("--no_plots", action="store_true", help="Skip plotting; CSVs still written.")

    # optional stress-profile switch (kept, but defaults already match mode_sep)
    p.add_argument("--stress_profile", type=str, default="mode_sep", choices=["none", "mode_sep"])

    args = p.parse_args()

    # Force multi-agent (project scope)
    args.multiagent = True
    if args.n_agents is None or int(args.n_agents) <= 0:
        args.n_agents = 5

    # Lock the final regime (even if someone edits defaults later)
    if args.stress_profile == "mode_sep":
        args.arrival_scale = 1600.0
        args.burst_k = 4.0
        args.device_skew = 1.35
        args.class_mix = (0.75, 0.20, 0.05)
        args.delay_penalty_w = 0.012
        args.energy_penalty_w = 0.00040
        args.sinr_sigma_db = 4.5
        args.device_jitter_cv = 0.25
        print(
            "[stress_profile=mode_sep] overrides applied:",
            f"arrival_scale={args.arrival_scale}, burst_k={args.burst_k}, "
            f"device_skew={args.device_skew}, class_mix={args.class_mix}, "
            f"delay_penalty_w={args.delay_penalty_w}, energy_penalty_w={args.energy_penalty_w}, "
            f"sinr_sigma_db={args.sinr_sigma_db}, device_jitter_cv={args.device_jitter_cv}",
        )

    # device
    if args.device == "auto":
        DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        DEV = torch.device("cuda") if args.device == "cuda" else torch.device("cpu")

    print(f"[device] using {DEV}")
    try:
        print(
            f"[cuda] available={torch.cuda.is_available()}  torch={torch.__version__}  "
            f"build_cuda={getattr(torch.version, 'cuda', None)}"
        )
        if torch.cuda.is_available():
            print(f"[cuda] device0={torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"[cuda] probe failed: {e}")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    ts = _now_ts()

    # Green-button paper run: always sweep all 3 modes in fixed order.
    modes = ["round_robin", "max_age", "max_weight"]
    print(f"[modes] {modes}")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        args.device_select_mode = mode

        outdir_mode = OUTDIR / f"{ts}_{mode}"
        _ensure_dir(outdir_mode)

        n_channels_ref = None
        all_runs: List[List[Tuple[str, Dict[str, Any]]]] = []
        conv_acc: Dict[str, List[int]] = defaultdict(list)
        time_acc: Dict[str, List[float]] = defaultdict(list)

        for sd in seeds:
            print(
                f"\n=== RUN mode={mode} seed={sd} n_devices={args.n_devices} "
                f"episodes={args.episodes} steps={args.steps} "
                f"multiagent=True n_agents={args.n_agents} ==="
            )

            run_results, train_time_s, conv_ep, n_channels = run_for_seed(args, sd, ts, DEV, outdir_mode)

            if n_channels_ref is None:
                n_channels_ref = int(n_channels)

            all_runs.append(run_results)

            for k, v in conv_ep.items():
                conv_acc[k].append(int(v))
            for k, v in train_time_s.items():
                time_acc[k].append(float(v))

            manifest = dict(
                seed=sd,
                mode=mode,
                n_devices=int(args.n_devices),
                n_channels=int(getattr(args, "n_channels", n_channels_ref or 1)),
                episodes=int(args.episodes),
                steps=int(args.steps),
                arrival_scale=float(args.arrival_scale),
                class_mix=tuple(args.class_mix),
                burst_k=float(args.burst_k),
                device_skew=float(args.device_skew),
                bits_scale=float(args.bits_scale),
                delay_penalty_w=float(args.delay_penalty_w),
                energy_penalty_w=float(args.energy_penalty_w),
                sinr_sigma_db=float(args.sinr_sigma_db),
                device_jitter_cv=float(args.device_jitter_cv),
                prio_w=tuple(args.prio_w),
                multiagent=True,
                n_agents=int(args.n_agents),
            )
            (outdir_mode / f"run_manifest_{ts}_seed{sd}.json").write_text(json.dumps(manifest, indent=2))

        # aggregate per mode
        agg_results = aggregate_over_seeds(all_runs)

        # summary per mode
        write_summary_csv(agg_results, ts, outdir_mode)

        # plots (optional, only if your utils.py supports them)
        if not args.no_plots and U is not None:
            try:
                if hasattr(U, "plot_reward_and_qvalue_convergence_across_seeds"):
                    U.plot_reward_and_qvalue_convergence_across_seeds(
                        all_runs, window=int(args.window), outdir=outdir_mode,
                        title="Convergence analysis (moving average)",
                    )

                if hasattr(U, "plot_cdf_bar_combo"):
                    for nm, m in agg_results:
                        eps = m.get("episode_metrics_std", []) or []
                        if not eps:
                            continue
                        m["throughput_per_episode"] = [float(r.get("throughput_mbps", 0.0)) for r in eps]
                        m["delay_per_episode"] = [float(r.get("avg_delay_ms", 0.0)) for r in eps]
                        m["energy_efficiency_per_episode"] = [float(r.get("energy_eff_bpj", 0.0)) for r in eps]
                        m["fairness_per_episode"] = [
                            float(r.get("jain_fairness_devices", r.get("jain_fairness", 0.0))) for r in eps
                        ]

                    U.plot_cdf_bar_combo(
                        agg_results, "throughput_per_episode",
                        "Throughput (Mbps)", "Throughput (Mbps)",
                        outdir=outdir_mode, fname="combo_throughput.png"
                    )
                    U.plot_cdf_bar_combo(
                        agg_results, "delay_per_episode",
                        "Delay (ms)", "Delay (ms)",
                        outdir=outdir_mode, fname="combo_delay.png"
                    )
                    U.plot_cdf_bar_combo(
                        agg_results, "energy_efficiency_per_episode",
                        "Energy Efficiency (bits/J)", "Energy Efficiency (bits/J)",
                        outdir=outdir_mode, fname="combo_energy_eff.png"
                    )
                    U.plot_cdf_bar_combo(
                        agg_results, "fairness_per_episode",
                        "Fairness", "Fairness",
                        outdir=outdir_mode, fname="combo_fairness.png"
                    )

                if hasattr(U, "plot_convergence_and_time"):
                    conv_mean = {k: int(np.mean(v)) if v else 0 for k, v in conv_acc.items()}
                    time_mean = {k: float(np.mean(v)) if v else 0.0 for k, v in time_acc.items()}
                    U.plot_convergence_and_time(conv_mean, time_mean, outdir=outdir_mode)

            except Exception as e:
                print(f"[WARN] plotting failed: {e.__class__.__name__}: {e}")
        else:
            if args.no_plots:
                print("[no_plots] Skipping plotting in sim_runner.py.")
            else:
                print("[plots] utils.py not available; skipping plots.")

        print(f"\n[MODE DONE] {mode} outputs in: {outdir_mode.resolve()}\n")

    print(f"\nAll done. Outputs in: {OUTDIR.resolve()}\n")

if __name__ == "__main__":
    main()
