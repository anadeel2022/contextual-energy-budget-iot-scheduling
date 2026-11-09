# rl_dqn.py
# Multi-agent only: (Centralized) DQN / Double DQN / Dueling Double DQN
# Optional CMDP-style constraints via primal-dual penalty on (blocking, interruption, energy)

from __future__ import annotations

from dataclasses import dataclass
from collections import deque, namedtuple
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import new_episode_acc, accumulate_from_info, finalize_episode_metrics, MetricKeys

Transition = namedtuple("Transition", ["s", "a", "r", "s2", "done"])


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)
def _flatten_ma_obs(obs: Any) -> np.ndarray:
    """
    Multi-agent env returns dict[int, obs_vec]. We do centralized learning by concatenating
    agent observations in sorted agent_id order.
    """
    if not isinstance(obs, dict):
        return np.asarray(obs, dtype=np.float32).reshape(1, -1)

    if not obs:
        return np.zeros((1, 0), dtype=np.float32)

    parts: List[np.ndarray] = []
    for aid in sorted(obs.keys()):
        oi = np.asarray(obs[aid], dtype=np.float32).reshape(1, -1)
        parts.append(oi)
    return np.concatenate(parts, axis=1).astype(np.float32)
def _ma_reward_to_scalar(r: Any, mode: str = "sum") -> float:
    """
    For multi-agent rewards dict {aid: r_i}, the only sane centralized default is SUM.
    """
    if not isinstance(r, dict):
        return _safe_float(r, 0.0)

    vals = []
    for v in r.values():
        vals.append(_safe_float(v, 0.0))
    if not vals:
        return 0.0
    if mode == "mean":
        return float(np.mean(vals))
    return float(np.sum(vals))
def _ma_done_to_bool(done: Any) -> bool:
    if isinstance(done, dict):
        if "__all__" in done:
            return bool(done["__all__"])
        return bool(all(bool(v) for v in done.values()))
    return bool(done)
def _format_broadcast_action(a: int, obs_dict: Optional[dict]) -> Any:
    """
    Centralized policy chooses a scalar action a, then broadcasts it to all agents.
    """
    if obs_dict is None:
        return int(a)
    return {aid: int(a) for aid in obs_dict.keys()}
def _signal_from_info(info: dict, primary: str, secondary: Optional[str] = None) -> float:
    """
    Extract scalar signal even if stored as dict/list/scalar.
    """
    v = info.get(primary, None)
    if v is None and secondary is not None:
        v = info.get(secondary, 0.0)

    if isinstance(v, dict):
        return float(np.sum([_safe_float(x, 0.0) for x in v.values()]))

    if isinstance(v, (list, tuple, np.ndarray)):
        return float(np.sum([_safe_float(x, 0.0) for x in v]))

    return _safe_float(v, 0.0)


class Replay:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=int(capacity))

    def push(self, *t):
        self.buf.append(Transition(*t))

    def sample(self, n: int):
        n = int(min(n, len(self.buf)))
        idx = np.random.choice(len(self.buf), size=n, replace=False)

        bs = np.vstack([self.buf[i].s for i in idx]).astype(np.float32)
        ba = np.asarray([self.buf[i].a for i in idx], dtype=np.int64)
        br = np.asarray([self.buf[i].r for i in idx], dtype=np.float32)
        bs2 = np.vstack([self.buf[i].s2 for i in idx]).astype(np.float32)
        bd = np.asarray([self.buf[i].done for i in idx], dtype=np.float32)

        return (
            torch.from_numpy(bs),
            torch.from_numpy(ba),
            torch.from_numpy(br),
            torch.from_numpy(bs2),
            torch.from_numpy(bd),
        )

    def __len__(self):
        return len(self.buf)


class QNet(nn.Module):
    def __init__(self, n_in: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DuelingQNet(nn.Module):
    def __init__(self, n_in: int, n_actions: int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.adv = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )
        self.val = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        f = self.feature(x)
        a = self.adv(f)
        v = self.val(f)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q


@dataclass
class DQNCfg:
    gamma: float = 0.99
    lr: float = 5e-4
    batch_size: int = 128
    buffer_size: int = 50_000
    target_sync: int = 200
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.995
    min_warmup_steps: int = 2000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    reward_agg: str = "sum"  # multi-agent default

    constrained: bool = False
    constr_lr: float = 1e-3
    constr_block_target: float | None = 0.02
    constr_intr_target: float | None = 0.01
    constr_energy_j_target: float | None = None


@dataclass
class DuelingCfg(DQNCfg):
    pass


class TorchPolicy:
    def __init__(self, net: nn.Module, device: torch.device):
        self.net = net
        self.device = device

    def act(self, obs: Any) -> int:
        x = _flatten_ma_obs(obs)
        xt = torch.from_numpy(x).float().to(self.device)
        with torch.no_grad():
            q = self.net(xt)
        return int(torch.argmax(q, dim=1).item())


def _train_centralized_dqn(
    env_ma,
    *,
    episodes: int,
    steps_per_ep: int,
    device: torch.device,
    cfg: DQNCfg,
    variant: str,  # "dqn" | "double" | "dueling"
) -> Dict[str, Any]:
    """
    Centralized learning: flatten multi-agent obs, broadcast action to all agents.
    For multi-agent reward dict, aggregate by cfg.reward_agg (default sum).
    """
    reset_out = env_ma.reset()
    obs0 = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    if not isinstance(obs0, dict):
        raise ValueError("This rl_dqn.py is multi-agent only, expected env.reset() -> dict obs.")

    s0 = _flatten_ma_obs(obs0)
    n_in = int(s0.shape[1])

    # IMPORTANT: action space is defined by the multi-agent wrapper (env_ma),
    # not by the internal SpectrumEnv instances.
    n_actions = int(getattr(env_ma, "n_actions", getattr(env_ma, "action_dim", 1)))

    if variant == "dueling":
        policy: nn.Module = DuelingQNet(n_in, n_actions).to(device)
        target: nn.Module = DuelingQNet(n_in, n_actions).to(device)
    else:
        policy = QNet(n_in, n_actions).to(device)
        target = QNet(n_in, n_actions).to(device)

    target.load_state_dict(policy.state_dict())
    opt = optim.Adam(policy.parameters(), lr=float(cfg.lr))
    buf = Replay(int(cfg.buffer_size))

    rewards_per_episode: List[float] = []
    q_value_max_raw: List[float] = []
    q_value_max: List[float] = []
    episode_metrics_std: List[dict] = []

    eps = float(cfg.eps_start)
    global_step = 0

    lam_block = 0.0
    lam_intr = 0.0
    lam_energy = 0.0
    lambda_block_hist: List[float] = []
    lambda_intr_hist: List[float] = []
    lambda_energy_hist: List[float] = []

    q_ema = None
    q_floor = float("inf")
    q_ceil = float("-inf")

    for ep in range(int(episodes)):
        reset_out = env_ma.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        obs_dict = obs if isinstance(obs, dict) else None

        s = _flatten_ma_obs(obs)
        ep_r_env = 0.0
        ep_r_train = 0.0  # shaped/clipped reward used for learning
        ep_qmax = -1e18

        ep_acc = new_episode_acc(getattr(env_ma, "n_classes", 3))

        for _t in range(int(steps_per_ep)):
            st = torch.from_numpy(s).float().to(device)
            with torch.no_grad():
                qvals = policy(st)

            if np.random.rand() < eps:
                a = int(np.random.randint(n_actions))
            else:
                a = int(torch.argmax(qvals, dim=1).item())

            ep_qmax = max(ep_qmax, _safe_float(qvals.max().item(), -1e18))

            action = _format_broadcast_action(a, obs_dict)

            s2_raw, env_r, done, info = env_ma.step(action)
            obs_dict2 = s2_raw if isinstance(s2_raw, dict) else None
            s2 = _flatten_ma_obs(s2_raw)

            env_r_scalar = _ma_reward_to_scalar(env_r, mode=str(cfg.reward_agg))
            done_bool = _ma_done_to_bool(done)

            blocked_step = _signal_from_info(info, "blocked", "deny")
            preempt_step = _signal_from_info(info, "preempt", "preempted")
            energy_step = _signal_from_info(info, "energy_j", "energy")

            penalty = 0.0
            if bool(cfg.constrained):
                penalty = lam_block * blocked_step + lam_intr * preempt_step + lam_energy * energy_step

            r = float(np.clip(env_r_scalar - penalty, -2.0, 2.0))
            ep_r_env += float(env_r_scalar)
            ep_r_train += float(r)

            buf.push(s, a, r, s2, float(done_bool))

            if len(buf) >= int(cfg.batch_size) and global_step >= int(cfg.min_warmup_steps):
                bs, ba, br, bs2, bd = buf.sample(int(cfg.batch_size))
                bs = bs.to(device)
                ba = ba.to(device)
                br = br.to(device)
                bs2 = bs2.to(device)
                bd = bd.to(device)

                with torch.no_grad():
                    if variant in {"double", "dueling"}:
                        # Double DQN target: argmax from online, value from target
                        next_q_online = policy(bs2)
                        next_actions = next_q_online.argmax(dim=1)
                        q_next = target(bs2).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    else:
                        # Vanilla DQN: max over target
                        q_next = target(bs2).max(dim=1).values
                    y = br + float(cfg.gamma) * (1.0 - bd) * q_next

                q_pred = policy(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(q_pred, y)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                opt.step()

                if global_step % int(cfg.target_sync) == 0:
                    target.load_state_dict(policy.state_dict())

            accumulate_from_info(ep_acc, info)

            s = s2
            obs_dict = obs_dict2
            global_step += 1

            if done_bool:
                break

        eps = max(float(cfg.eps_end), eps * float(cfg.eps_decay))
        rewards_per_episode.append(_safe_float(ep_r_env, 0.0))

        # Q diagnostics: raw EMA, plus normalized monotone score for convergence plots
        alpha = 0.15
        q_ema = _safe_float(ep_qmax, 0.0) if q_ema is None else (alpha * _safe_float(ep_qmax, 0.0) + (1.0 - alpha) * q_ema)
        q_floor = min(q_floor, q_ema)
        q_ceil = max(q_ceil, q_ema)

        score = float((q_ema - q_floor) / ((q_ceil - q_floor) + 1e-9))
        if q_value_max:
            score = max(score, float(q_value_max[-1]))
        q_value_max.append(score)
        q_value_max_raw.append(float(q_ema))

        m = finalize_episode_metrics(
            env_ma,
            steps_per_ep=int(steps_per_ep),
            num_channels=int(getattr(env_ma, "n_channels", 1)),
            slot_s=float(getattr(env_ma, "slot_s", 0.001)),
            ep_acc=ep_acc,
        )
        # Convergence signals for plotting
        m["episode"] = int(ep)
        m["episode_return"] = float(ep_r_env)  # keep for backward compatibility with your plotter
        m["episode_return_env"] = float(ep_r_env)  # explicit
        m["episode_return_train"] = float(ep_r_train)  # explicit (shaped/clipped)
        m[MetricKeys.QMAX] = float(ep_qmax) if np.isfinite(ep_qmax) else 0.0

        # primal–dual updates (episode-level signals)
        if bool(cfg.constrained) and float(cfg.constr_lr) > 0.0:
            blk_val = m.get(MetricKeys.BLOCK, np.nan)
            intr_val = m.get(MetricKeys.INTR, np.nan)
            energy_val = m.get(MetricKeys.ENERGY_J, np.nan)

            if cfg.constr_block_target is not None and np.isfinite(blk_val):
                lam_block = max(0.0, lam_block + float(cfg.constr_lr) * (float(blk_val) - float(cfg.constr_block_target)))
            if cfg.constr_intr_target is not None and np.isfinite(intr_val):
                lam_intr = max(0.0, lam_intr + float(cfg.constr_lr) * (float(intr_val) - float(cfg.constr_intr_target)))
            if cfg.constr_energy_j_target is not None and np.isfinite(energy_val):
                lam_energy = max(0.0, lam_energy + float(cfg.constr_lr) * (float(energy_val) - float(cfg.constr_energy_j_target)))

        lambda_block_hist.append(float(lam_block))
        lambda_intr_hist.append(float(lam_intr))
        lambda_energy_hist.append(float(lam_energy))

        episode_metrics_std.append(m)

    out: Dict[str, Any] = {
        "rewards_per_episode": rewards_per_episode,
        "q_value_max": q_value_max,
        "q_value_max_raw": q_value_max_raw,
        "episode_metrics_std": episode_metrics_std,
        "policy": TorchPolicy(policy, device),
        "lambda_block": lambda_block_hist,
        "lambda_intr": lambda_intr_hist,
        "lambda_energy": lambda_energy_hist,
    }

    try:
        eps_std = out.get("episode_metrics_std", []) or []
        if eps_std:
            def _series(key: str):
                return [float(mm.get(key, float("nan"))) for mm in eps_std]

            out["throughput_per_episode"] = _series("throughput_mbps")
            out["delay_per_episode"] = _series("avg_delay_ms")
            out["energy_usage_per_episode"] = _series("energy_j")
            out["energy_efficiency_per_episode"] = _series("energy_eff_bpj")
            out["fairness_per_episode"] = _series("jain_fairness_devices")
            out["blocking_prob_per_episode"] = _series("blocking_prob")
            out["interrupt_prob_per_episode"] = _series("interrupt_prob")
            out["utilization_per_episode"] = _series("utilization")

    except Exception:
        pass

    return out


def rl_dqn_multiagent_train(
    env_ma=None,
    *,
    episodes: int = 500,
    steps_per_ep: int = 1000,
    device: Any = None,
    cfg: Optional[DQNCfg] = None,
    constrained: Optional[bool] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Entry point expected by sim_runner.py for "DQN".
    This implementation is true VANILLA DQN target (max over target net).
    """
    env = env_ma if env_ma is not None else kwargs.get("env", None)
    if env is None:
        raise ValueError("RL_dqn_multiagent_train: env_ma (or env) must be provided.")

    cfg = cfg or DQNCfg()

    if constrained is not None:
        cfg.constrained = bool(constrained)

    # If runner passes constraint targets directly, map into cfg
    for k in ["constr_block_target", "constr_intr_target", "constr_energy_j_target", "constr_lr"]:
        if k in kwargs and hasattr(cfg, k):
            setattr(cfg, k, kwargs[k])

    if device is None or device == "auto":
        dev = torch.device(str(cfg.device))
    elif isinstance(device, str):
        dev = torch.device(device)
    else:
        dev = device

    return _train_centralized_dqn(
        env,
        episodes=int(episodes),
        steps_per_ep=int(steps_per_ep),
        device=dev,
        cfg=cfg,
        variant="dqn",
    )


def rl_double_dqn_multiagent_train(
    env_ma=None,
    *,
    episodes: int = 500,
    steps_per_ep: int = 1000,
    device: Any = None,
    cfg: Optional[DQNCfg] = None,
    constrained: Optional[bool] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Explicit Double DQN entry point (optional).
    """
    env = env_ma if env_ma is not None else kwargs.get("env", None)
    if env is None:
        raise ValueError("RL_double_dqn_multiagent_train: env_ma (or env) must be provided.")

    cfg = cfg or DQNCfg()

    if constrained is not None:
        cfg.constrained = bool(constrained)

    for k in ["constr_block_target", "constr_intr_target", "constr_energy_j_target", "constr_lr"]:
        if k in kwargs and hasattr(cfg, k):
            setattr(cfg, k, kwargs[k])

    if device is None or device == "auto":
        dev = torch.device(str(cfg.device))
    elif isinstance(device, str):
        dev = torch.device(device)
    else:
        dev = device

    return _train_centralized_dqn(
        env,
        episodes=int(episodes),
        steps_per_ep=int(steps_per_ep),
        device=dev,
        cfg=cfg,
        variant="double",
    )


def rl_dueling_dqn_multiagent_train(
    env_ma=None,
    *,
    episodes: int = 500,
    steps_per_ep: int = 1000,
    device: Any = None,
    cfg: Optional[DuelingCfg] = None,
    constrained: Optional[bool] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Entry point expected by sim_runner.py for "DuelingDQN".
    Uses Dueling architecture with Double-DQN target.
    """
    env = env_ma if env_ma is not None else kwargs.get("env", None)
    if env is None:
        raise ValueError("RL_dueling_dqn_multiagent_train: env_ma (or env) must be provided.")

    cfg = cfg or DuelingCfg()

    if constrained is not None:
        cfg.constrained = bool(constrained)

    for k in ["constr_block_target", "constr_intr_target", "constr_energy_j_target", "constr_lr"]:
        if k in kwargs and hasattr(cfg, k):
            setattr(cfg, k, kwargs[k])

    if device is None or device == "auto":
        dev = torch.device(str(cfg.device))
    elif isinstance(device, str):
        dev = torch.device(device)
    else:
        dev = device

    return _train_centralized_dqn(
        env,
        episodes=int(episodes),
        steps_per_ep=int(steps_per_ep),
        device=dev,
        cfg=cfg,
        variant="dueling",
    )
