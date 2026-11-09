# ============================================================
# utils.py — FINAL (MULTI-AGENT ONLY)
# ============================================================
from __future__ import annotations
import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# GLOBAL CONSTANTS (fallbacks)
# ============================================================
SLOT_DURATION_S = 1e-3
CHANNEL_RATE_BPS = 10e6

# ----------------------------
# Numeric helpers (utils.py)
# ----------------------------
EPS = 1e-12
@dataclass(frozen=True)
class MetricKeys:
    # Episode totals / KPIs (column names)
    BITS_TX = "bits_tx_ep"
    THR_Mbps = "throughput_mbps"

    # Delay key: keep BOTH spellings for backward compatibility
    DELAY_MS = "avg_delay_ms"
    DELAY_ms = "avg_delay_ms"   # alias (older code path)

    BLOCK = "blocking_prob"
    INTR = "interrupt_prob"
    ENERGY_J = "energy_j"
    EE_BPJ = "energy_eff_bpj"
    UTIL = "utilization"

    # Q statistic key used by your plotter/convergence
    QMAX = "q_value_max"

    # Fairness
    FAIR = "jain_fairness"                     # keep legacy name
    FAIR_DEV = "jain_fairness_devices"
    FAIR_CLS = "jain_fairness_classes"


# ============================================================
# BASIC HELPERS
# ============================================================
def safe_div(a, b, default: float = 0.0) -> float:
    """
    Stable scalar division with finite checks and EPS guard.
    Always returns a Python float.
    """
    try:
        a = float(a)
        b = float(b)
        if (not np.isfinite(a)) or (not np.isfinite(b)) or abs(b) < EPS:
            return float(default)
        return float(a / b)
    except Exception:
        return float(default)
def jain_index(x) -> float:
    """
    Jain's fairness index for a nonnegative vector x.
    Returns 0.0 if x sums to ~0, and clamps into [0, 1].
    """
    v = np.asarray(x, dtype=float).reshape(-1)
    v[~np.isfinite(v)] = 0.0
    s1 = float(np.sum(v))
    if s1 <= EPS:
        return 0.0
    s2 = float(np.sum(v * v))
    if s2 <= EPS:
        return 0.0
    n = float(v.size)
    j = (s1 * s1) / (n * s2)
    if not np.isfinite(j):
        return 0.0
    return float(np.clip(j, 0.0, 1.0))

def _ts():
    return time.strftime("%Y%m%d_%H%M%S")

def new_episode_acc(n_classes: int = 3, n_devices: int | None = None):
    """
    Episode accumulator that supports both class- and device-level fairness.
    n_devices can be provided (recommended). If None, device vector is lazily created.
    """
    acc = {
        "steps": 0,
        "bits": 0.0,
        "delays_ms": [],
        "arrivals": 0,
        "blocked": 0,
        "preempted": 0,
        "scheduled": 0,
        "slots_used": 0,
        "energy_j": 0.0,
        "served_bits_per_class": np.zeros(int(n_classes), dtype=float),
        "served_bits_per_device": None,
    }
    if n_devices is not None:
        acc["served_bits_per_device"] = np.zeros(int(n_devices), dtype=float)
    return acc

def accumulate_from_info(acc: dict, info: dict, n_classes: int = 3):
    """
    Robust accumulator:
    - accepts single-agent info dict
    - accepts multi-agent info dict {aid: info_dict}
    - tolerates nested wrappers ({"info": {...}}, {"infos": {...}})
    """
    if info is None:
        return

    # Unwrap common wrappers
    if isinstance(info, dict) and "info" in info and isinstance(info["info"], dict):
        info = info["info"]
    if isinstance(info, dict) and "infos" in info and isinstance(info["infos"], dict):
        info = info["infos"]

    if not isinstance(info, dict) or not info:
        return

    # Multi-agent recurse (must be top-level and must end here)
    if all(isinstance(v, dict) for v in info.values()):
        for v in info.values():
            accumulate_from_info(acc, v, n_classes=n_classes)
        return

    # ---- Single-agent info below ----
    acc.setdefault("steps", 0)
    acc.setdefault("arrivals", 0)
    acc.setdefault("blocked", 0)
    acc.setdefault("preempted", 0)
    acc.setdefault("scheduled", 0)
    acc.setdefault("slots_used", 0)
    acc.setdefault("bits", 0.0)
    acc.setdefault("delivered_bits", 0.0)
    acc.setdefault("energy_j", 0.0)
    acc.setdefault("delays_ms", [])
    acc.setdefault("served_bits_per_class", np.zeros(int(n_classes), dtype=float))

    acc["steps"] += 1

    acc["arrivals"] += int(info.get("arrivals", 0) or 0)
    acc["blocked"] += int(info.get("blocked", info.get("denies", info.get("deny", 0))) or 0)
    acc["preempted"] += int(info.get("preempted", info.get("preempt", 0)) or 0)
    acc["scheduled"] += int(info.get("scheduled", 0) or 0)
    acc["slots_used"] += int(info.get("slots_used", 0) or 0)

    # Bits delivered (aliases)
    delivered = None
    for k in ("delivered_bits", "bits_delivered", "bits_tx", "bits"):
        if k in info:
            try:
                v = float(info.get(k, 0.0))
                if np.isfinite(v):
                    delivered = v
            except Exception:
                delivered = None
            break

    if delivered is not None:
        acc["bits"] += delivered
        acc["delivered_bits"] += delivered

    # Energy
    if "energy_j" in info:
        try:
            v = float(info.get("energy_j", 0.0))
            if np.isfinite(v):
                acc["energy_j"] += v
        except Exception:
            pass
    elif "energy_mj" in info:
        try:
            v = float(info.get("energy_mj", 0.0)) * 1e-3
            if np.isfinite(v):
                acc["energy_j"] += v
        except Exception:
            pass

    # Delay
    d = info.get("delay_ms", None)
    if d is not None:
        try:
            v = float(d)
            if np.isfinite(v) and v >= 0:
                acc["delays_ms"].append(v)
        except Exception:
            pass

    # Served bits per class (delta vector)
    cls_bits = info.get("served_bits_per_class", None)
    if cls_bits is not None:
        v = np.asarray(cls_bits, dtype=float).ravel()
        v[~np.isfinite(v)] = 0.0

        if acc["served_bits_per_class"].size != int(n_classes):
            acc["served_bits_per_class"] = np.zeros(int(n_classes), dtype=float)

        if v.size != acc["served_bits_per_class"].size:
            w = np.zeros_like(acc["served_bits_per_class"])
            m = min(v.size, w.size)
            w[:m] = v[:m]
            v = w

        acc["served_bits_per_class"] += v

    # Served bits per device (delta vector)
    dev_bits = info.get("served_bits_per_device", None)
    if dev_bits is not None:
        v = np.asarray(dev_bits, dtype=float).ravel()
        v[~np.isfinite(v)] = 0.0

        if acc.get("served_bits_per_device", None) is None:
            acc["served_bits_per_device"] = np.zeros(v.size, dtype=float)

        if v.size != acc["served_bits_per_device"].size:
            w = np.zeros_like(acc["served_bits_per_device"])
            m = min(v.size, w.size)
            w[:m] = v[:m]
            v = w

        acc["served_bits_per_device"] += v

def finalize_episode_metrics(
    env,
    steps_per_ep: int,
    num_channels: int,
    slot_s: float,
    ep_acc: dict,
) -> dict:
    steps = int(ep_acc.get("steps", 0)) or int(steps_per_ep)

    # Prefer passed values; fall back to env attributes; then constants.
    slot_s = float(slot_s or getattr(env, "slot_s", 1e-3))
    num_channels = int(num_channels or getattr(env, "n_channels", 1))
    ch_rate = float(getattr(env, "channel_rate_bps", 1e8))

    arrivals = int(ep_acc.get("arrivals", 0))
    blocked = int(ep_acc.get("blocked", 0))
    preempted = int(ep_acc.get("preempted", 0))
    scheduled = int(ep_acc.get("scheduled", 0))
    slots_used = int(ep_acc.get("slots_used", 0))

    bits = float(ep_acc.get("delivered_bits", ep_acc.get("bits", 0.0)))
    energy = float(ep_acc.get("energy_j", 0.0))

    thr_mbps = safe_div(bits, steps * slot_s) / 1e6
    delay = float(np.mean(ep_acc["delays_ms"])) if ep_acc.get("delays_ms") else 0.0
    ee = safe_div(bits, energy)

    cap_bits = max(steps * num_channels, 1) * ch_rate * slot_s
    util = safe_div(bits, cap_bits)

    block_prob = safe_div(blocked, max(arrivals, 1))
    intr_prob = safe_div(preempted, max(scheduled, 1))

    served_bits_per_class = np.asarray(ep_acc.get("served_bits_per_class", np.zeros(3)), dtype=float).reshape(-1)
    served_bits_per_class[~np.isfinite(served_bits_per_class)] = 0.0

    served_bits_per_device = ep_acc.get("served_bits_per_device", None)
    if served_bits_per_device is None:
        served_bits_per_device = np.zeros(int(getattr(env, "n_devices", 1)), dtype=float)
    served_bits_per_device = np.asarray(served_bits_per_device, dtype=float).reshape(-1)
    served_bits_per_device[~np.isfinite(served_bits_per_device)] = 0.0

    fairness_devices = jain_index(served_bits_per_device)
    fairness_classes = jain_index(served_bits_per_class)

    # Selection diagnostics (aggregate over internal agents if present)
    sel_entropy = 0.0
    sel_top = 0.0
    sel_active = 0
    try:
        if hasattr(env, "agents") and isinstance(env.agents, dict) and env.agents:
            counts = None
            for _aid, aenv in env.agents.items():
                c = getattr(aenv, "selected_dev_counts_ep", None)
                if c is None:
                    continue
                c = np.asarray(c, dtype=float).reshape(-1)
                counts = c if counts is None else (counts + c)

            if counts is not None:
                tot = float(np.sum(counts))
                if tot > 0:
                    p = counts / tot
                    p_pos = p[p > 0]
                    sel_entropy = float(-np.sum(p_pos * np.log(p_pos + EPS)))
                    sel_top = float(np.max(p))
                    sel_active = int(np.sum(counts > 0))
    except Exception:
        sel_entropy, sel_top, sel_active = 0.0, 0.0, 0

    # ------------------------------------------------------------
    # Per-class served bits and fractions (for CSV + plots)  [D]
    # ------------------------------------------------------------
    served_cls = np.asarray(ep_acc.get("served_bits_per_class", np.zeros(3)), dtype=float).reshape(-1)
    served_cls[~np.isfinite(served_cls)] = 0.0
    served_cls_sum = float(np.sum(served_cls))
    served_frac = (served_cls / served_cls_sum) if served_cls_sum > EPS else np.zeros_like(served_cls)

    out = {
        MetricKeys.THR_Mbps: float(thr_mbps),
        MetricKeys.DELAY_MS: float(delay),
        MetricKeys.EE_BPJ: float(ee),
        MetricKeys.FAIR: float(fairness_devices),  # keep FAIR as device fairness
        "jain_fairness_devices": float(fairness_devices),
        "jain_fairness_classes": float(fairness_classes),
        MetricKeys.BLOCK: float(block_prob),
        MetricKeys.INTR: float(intr_prob),
        MetricKeys.UTIL: float(util),

        "sel_entropy": float(sel_entropy),
        "sel_top_share": float(sel_top),
        "sel_active_devices": int(sel_active),

        # raw totals for integrity + multi-seed
        "arrivals": int(arrivals),
        "blocked": int(blocked),
        "preempted": int(preempted),
        "scheduled": int(scheduled),
        "slots_used": int(slots_used),
        "bits": float(bits),
        "energy_j": float(energy),
    }

    for k in range(min(int(served_cls.size), 16)):
        out[f"served_bits_class{k}"] = float(served_cls[k])
        out[f"served_frac_class{k}"] = float(served_frac[k])

    return out
# ============================================================
# CSV WRITER
# ============================================================
def save_metrics_csv(path: Path, metrics: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not metrics:
        # still write header-less empty file if you want; otherwise return
        return

    # Union of keys across all rows
    keys = sorted({k for m in metrics if isinstance(m, dict) for k in m.keys()})

    def _clean(v):
        # normalize numpy scalars
        try:
            if isinstance(v, (np.generic,)):
                v = v.item()
        except Exception:
            pass

        # map NaN/inf to 0.0 to avoid blank-looking spreadsheet cells
        if isinstance(v, float):
            if (not np.isfinite(v)) or np.isnan(v):
                return 0.0
            return float(v)

        # ints/bools are fine
        if isinstance(v, (int, bool)):
            return int(v)

        # if missing or None, write 0.0
        if v is None:
            return 0.0

        return v  # strings or other types

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in metrics:
            if not isinstance(row, dict):
                row = {}
            out = {k: _clean(row.get(k, 0.0)) for k in keys}
            w.writerow(out)

# ============================================================
# PLOTTING (MULTI-AGENT)
# ============================================================
def plot_mean_bar(agg_results, key, ylabel, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    names, means, stds = [], [], []
    for name, m in agg_results:
        vals = np.asarray([r.get(key, np.nan) for r in m["episode_metrics_std"]])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        names.append(name)
        means.append(vals.mean())
        stds.append(vals.std())

    if not names:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    ax.bar(x, means)
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"bar_{key}.png", dpi=300)
    plt.close(fig)
