"""Causal candidate-aware 3-action energy CMDP comparison (exploratory).

Important: this changes environment decision timing for ALL methods. It is not
reproduction of v6 results and must not be pooled with them.
"""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
import numpy as np
import torch
import energy_cmdp as core
from energy_observation_v5 import ORIGINAL_ENV_FOR
from three_action_v6 import write, summary
from three_action_v6 import neural_eval as _v6_neural_eval


def env_for(seed, cfg):
    env=ORIGINAL_ENV_FOR(seed,cfg)
    env.n_actions=env.action_dim=3
    for unit in env.agents.values():
        unit.three_action_enabled=True
        unit.energy_features_enabled=True
        unit.predecision_candidate_enabled=True
        unit.energy_obs_budget_j=cfg.energy_limit_j_per_step or 0.0002944864377500498
        unit.low_power_backoff_db=6.0
    return env

core.env_for=env_for
_v6_neural_eval.__globals__["env_for"] = env_for
neural_eval=_v6_neural_eval


def energy_aware_action(unit, cfg, remaining_steps, margin=0.):
    """Causal, budget-rate tracker using nominal mode energy and channel score.

    No future instantaneous fades, success outcomes or evaluation statistics.
    Allow normal power primarily on favorable large-scale links when the
    budget is ahead of its linear consumption schedule.
    """
    i=int(unit._pending_device)
    if unit.queue_bits[i] <= 0: return 2
    budget=float(cfg.energy_limit_j_per_step)*unit.steps_per_ep
    used=float(unit.energy_j_ep)
    left=max(unit.steps_per_ep-unit.t,1)
    allowance=max(0.,(budget-used)/left)
    w=float(10**((unit.tx_power_dbm[i]-30)/10))
    normal=(w+unit.circuit_power_w)*unit.slot_s + unit.energy_per_bit[i]*unit.bits_per_sample[i]
    low=(w*10**(-6/10)+unit.circuit_power_w)*unit.slot_s + unit.energy_per_bit[i]*unit.bits_per_sample[i]*0.7
    good_link=float(unit.mean_sinr_db[i]+unit.shadow_db_ep[i]) >= 8.
    # Conservative pacing, since successful TX adds processing energy.
    if good_link and normal <= allowance*(1+margin): return 1
    return 2


def baseline_eval(seed,cfg,method):
    env=env_for(1_000_000+seed,cfg); ids=list(env.agents); rows=[]
    for ep in range(cfg.eval_episodes):
        obs=env.reset(); bits=energy=0.; denied=eligible=low=tx=0
        for t in range(cfg.steps):
            actions={}
            for i,u in env.agents.items():
                if method=='BudgetAware': a=energy_aware_action(u,cfg,cfg.steps-t)
                elif method=='AlwaysNormal':a=1
                elif method=='AlwaysLow':a=2
                elif method=='Mix50':a=2 if t%2==0 else 1
                elif method=='Mix75':a=2 if t%4 else 1
                elif method=='ChannelThreshold':a=1 if u.mean_sinr_db[u._pending_device]+u.shadow_db_ep[u._pending_device]>=9 else 2
                else:raise ValueError(method)
                actions[i]=a
            obs,_,done,infos=env.step(actions)
            for i in ids:
                v=infos[i];bits+=float(v['delivered_bits']);energy+=float(v['energy_j']);denied+=int(v['blocked']);eligible+=int(v['blocked']+v['scheduled']);low+=int(v['tx_mode']=='low' and v['scheduled']);tx+=int(v['scheduled'])
            if done:break
        rows.append(dict(method=method,seed=seed,episode=ep+1,throughput_mbps=bits/(len(ids)*cfg.steps*env.slot_s*1e6),energy_mj_step=1000*energy/(len(ids)*cfg.steps),denied=denied,eligible=eligible,low_fraction=low/max(tx,1)))
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',default='201,202,203');p.add_argument('--episodes',type=int,default=8)
    p.add_argument('--eval-episodes',type=int,default=4);p.add_argument('--steps',type=int,default=50)
    p.add_argument('--cap-mj',type=float,default=.2944864377500498)
    p.add_argument('--out',default='v7_smoke');p.add_argument('--threads',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(a.threads)
    cfg=core.RunConfig(episodes=a.episodes,eval_episodes=a.eval_episodes,steps=a.steps,service_logit_bias=2.,denial_limit=.02,energy_limit_j_per_step=a.cap_mj/1000)
    folder=Path(a.out);folder.mkdir(parents=True,exist_ok=True)
    methods=('AlwaysNormal','AlwaysLow','Mix50','Mix75','ChannelThreshold','BudgetAware')
    records=[]; history=[]
    for seed in map(int,a.seeds.split(',')):
        for method in methods:records+=baseline_eval(seed,cfg,method)
        for constrained in (False,True):
            name='PD-PPO' if constrained else 'PPO';print('TRAIN',name,seed,flush=True)
            model,h=core.train(seed,cfg,constrained);history+=h
            records+=neural_eval(model,seed,cfg,name)
            write(folder/'evaluation_episodes.csv',records)
            write(folder/'per_seed_partial.csv',summary(records,a.cap_mj,cfg.denial_limit))
    per=summary(records,a.cap_mj,cfg.denial_limit)
    write(folder/'evaluation_episodes.csv',records);write(folder/'per_seed.csv',per)
    write(folder/'training_diagnostics.csv',history)
    out=[]
    for method in sorted(set(r['method'] for r in per)):
        x=[r for r in per if r['method']==method]
        out.append(dict(method=method,n=len(x),throughput_mbps=float(np.mean([r['throughput_mbps'] for r in x])),energy_mj_step=float(np.mean([r['energy_mj_step'] for r in x])),denial=float(np.mean([r['denial'] for r in x])),feasible=sum(r['joint_feasible'] for r in x)))
    write(folder/'summary.csv',out)
    (folder/'manifest.json').write_text(json.dumps(dict(status='exploratory only',seeds=a.seeds,train_episodes=a.episodes,eval_episodes=a.eval_episodes,cap_mj=a.cap_mj,denial_limit=cfg.denial_limit,note='v6 energy cap reused for initial diagnostic; recalibrate after decision-timing change before confirmatory testing'),indent=2))
    print('SUMMARY',json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
