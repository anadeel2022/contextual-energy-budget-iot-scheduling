"""
rl_tabular.py

Unified tabular RL for MultiAgentSpectrumEnv:
- Q-learning
- Double Q-learning

Multi-agent ONLY.
No single-agent adapters.
No legacy compatibility code.

Author: finalized for MARL-IEEE-ITS
"""

from __future__ import annotations
import time
import numpy as np
from typing import Dict, Any

from utils import (
    new_episode_acc,
    accumulate_from_info,
    finalize_episode_metrics,
    MetricKeys,
)


# ============================================================
# Helpers
# ============================================================

def _epsilon_greedy(q_row: np.ndarray, eps: float) -> int:
    if np.random.rand() < eps:
        return np.random.randint(len(q_row))
    return int(np.argmax(q_row))


def _ensure_state(Q: Dict, s, n_actions: int):
    if s not in Q:
        Q[s] = np.zeros(n_actions, dtype=float)


# ============================================================
# Core unified trainer
# ============================================================

def _train_tabular(
    env,
    *,
    episodes: int,
    steps_per_ep: int,
    alpha: float,
    gamma: float,
    eps_start: float,
    eps_end: float,
    eps_decay: float,
    double_q: bool = False,
) -> Dict[str, Any]:
    """
    Unified Q / Double-Q training loop for MultiAgentSpectrumEnv (multi-agent only).
    """

    # In this framework, actions correspond to selecting a device index
    n_actions = int(env.n_devices)
    assert n_actions > 0, "n_actions must be positive"

    Q1: Dict[Any, np.ndarray] = {}
    Q2: Dict[Any, np.ndarray] = {} if double_q else None

    rewards_per_episode = []
    episode_metrics_std = []

    eps = float(eps_start)

    for ep in range(int(episodes)):
        obs = env.reset()  # dict: {agent_id: obs}
        ref_env = next(iter(env.agents.values()))
        ep_acc = new_episode_acc(ref_env.n_classes)

        total_reward_train = 0.0
        total_reward_env = 0.0

        for t in range(int(steps_per_ep)):
            # ---- Encode global state ----
            s = env.encode_state(obs)
            if not isinstance(s, (int, tuple)):
                s = tuple(np.asarray(s).flatten())

            _ensure_state(Q1, s, n_actions)
            if double_q:
                _ensure_state(Q2, s, n_actions)

            # ---- Action selection ----
            if double_q:
                q_sum = Q1[s] + Q2[s]
                a = _epsilon_greedy(q_sum, eps)
            else:
                a = _epsilon_greedy(Q1[s], eps)

            # ---- Multi-agent step (dict actions REQUIRED) ----

            actions = {aid: a for aid in env.agents.keys()}
            obs_next, rewards, done, info = env.step(actions)

            # Aggregate reward across agents (mean is consistent with PPO/DQN)
            if isinstance(rewards, dict) and rewards:
                r = float(np.mean(list(rewards.values())))
            else:
                r = float(rewards) if rewards is not None else 0.0

            # ---- Next state ----
            s_next = env.encode_state(obs_next)
            if not isinstance(s_next, (int, tuple)):
                s_next = tuple(np.asarray(s_next).flatten())

            _ensure_state(Q1, s_next, n_actions)
            if double_q:
                _ensure_state(Q2, s_next, n_actions)

            # ---- Q update ----
            if double_q:
                if np.random.rand() < 0.5:
                    a_star = int(np.argmax(Q1[s_next]))
                    td_target = r + gamma * Q2[s_next][a_star]
                    Q1[s][a] += alpha * (td_target - Q1[s][a])
                else:
                    a_star = int(np.argmax(Q2[s_next]))
                    td_target = r + gamma * Q1[s_next][a_star]
                    Q2[s][a] += alpha * (td_target - Q2[s][a])
            else:
                td_target = r + gamma * np.max(Q1[s_next])
                Q1[s][a] += alpha * (td_target - Q1[s][a])

            total_reward_train += r
            total_reward_env += float(np.mean(list(rewards.values()))) if isinstance(rewards, dict) else float(r)

            # ---- Standardized metrics accumulation ----
            accumulate_from_info(ep_acc, info)

            obs = obs_next
            if done:
                break

        # ---- Episode finalize ----
        rewards_per_episode.append(float(total_reward_train))

        ref_env = next(iter(env.agents.values()))
        ep_metrics = finalize_episode_metrics(
            env=env,
            steps_per_ep=int(steps_per_ep),
            num_channels=int(ref_env.n_channels),
            slot_s=float(ref_env.slot_s),
            ep_acc=ep_acc,
        )

        # Convergence signals for plotting
        ep_metrics["episode_return_train"] = float(total_reward_train)
        ep_metrics["episode_return_env"] = float(total_reward_env)
        ep_metrics["episode"] = int(ep)
        ep_metrics["episode_return"] = float(total_reward_env)  # compatibility with convergence plotters

        # Q statistic for tabular: global max Q over all seen states
        if double_q:
            qmax = 0.0
            if Q1:
                qmax = max(qmax, float(max(np.max(v) for v in Q1.values())))
            if Q2:
                qmax = max(qmax, float(max(np.max(v) for v in Q2.values())))
            ep_metrics[MetricKeys.QMAX] = float(qmax)
        else:
            qmax = float(max(np.max(v) for v in Q1.values())) if Q1 else 0.0
            ep_metrics[MetricKeys.QMAX] = float(qmax)

        episode_metrics_std.append(ep_metrics)

        eps = max(float(eps_end), eps * float(eps_decay))

    return {
        "rewards_per_episode": rewards_per_episode,
        "episode_metrics_std": episode_metrics_std,
    }



# ============================================================
# Public API
# ============================================================

def rl_qlearning_multiagent_train(
    env,
    *,
    episodes: int,
    steps_per_ep: int,
    alpha: float = 0.2,
    gamma: float = 0.95,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: float = 0.995,
) -> Dict[str, Any]:
    """
    Standard Q-learning (multi-agent).
    """
    return _train_tabular(
        env,
        episodes=episodes,
        steps_per_ep=steps_per_ep,
        alpha=alpha,
        gamma=gamma,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
        double_q=False,
    )


def rl_doubleq_multiagent_train(
    env,
    *,
    episodes: int,
    steps_per_ep: int,
    alpha: float = 0.2,
    gamma: float = 0.95,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: float = 0.995,
) -> Dict[str, Any]:
    """
    Double Q-learning (multi-agent).
    """
    return _train_tabular(
        env,
        episodes=episodes,
        steps_per_ep=steps_per_ep,
        alpha=alpha,
        gamma=gamma,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
        double_q=True,
    )
