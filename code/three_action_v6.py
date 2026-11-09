"""Pilot-only three-action energy-aware CMDP; never tune thresholds on test seeds.
Action 0 deny, 1 normal power, 2 low power. Lower radiated power lowers SINR.
No change to the legacy binary simulator unless explicitly enabled.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
import energy_cmdp as core
from energy_observation_v5 import ORIGINAL_ENV_FOR


def env_for(seed,cfg):
    env=ORIGINAL_ENV_FOR(seed,cfg)
    env.n_actions=env.action_dim=3
    for e in env.agents.values():
        e.three_action_enabled=True
        e.energy_features_enabled=True
        e.energy_obs_budget_j=cfg.energy_limit_j_per_step or .000343
        e.low_power_backoff_db=6.0
    return env
core.env_for=env_for


def fixed_eval(seed,cfg,method):
    env=env_for(1_000_000+seed,cfg)
    ids=list(env.agents); rows=[]
    for episode in range(cfg.eval_episodes):
        obs=env.reset(); b=en=0.;den=elig=low=attempts=0
        for t in range(cfg.steps):
            acts={}
            for i,u in env.agents.items():
                if method=='AlwaysNormal': a=1
                elif method=='AlwaysLow': a=2
                elif method=='Periodic2pct': a=0 if (t+1)%50==0 else 1
                elif method=='Mix50': a=2 if t%2==0 else 1
                elif method=='Mix75': a=2 if t%4!=0 else 1
                elif method=='QueueUrgency':
                    # simple causal heuristic: normal for older queued packets, low otherwise
                    ages=obs[i][cfg.devices:2*cfg.devices]
                    a=1 if np.max(ages)>=0.20 else 2
                else:raise ValueError(method)
                acts[i]=a
            obs,_,done,infos=env.step(acts)
            for i in ids:
                info=infos[i]; b+=float(info['delivered_bits']);en+=float(info['energy_j']); den+=int(info['blocked']);elig+=int(info['blocked']+info['scheduled']);low+=int(info['tx_mode']=='low' and info['scheduled']);attempts+=int(info['scheduled'])
            if done:break
        rows.append(dict(method=method,seed=seed,episode=episode+1,throughput_mbps=b/(len(ids)*cfg.steps*env.slot_s*1e6),energy_mj_step=1000*en/(len(ids)*cfg.steps),denied=den,eligible=elig,low_fraction=low/max(attempts,1)))
    return rows


def neural_eval(model,seed,cfg,method):
    env=env_for(1_000_000+seed,cfg);ids=list(env.agents);rows=[]
    model.eval()
    with torch.no_grad():
        for episode in range(cfg.eval_episodes):
            obs=env.reset();b=en=0.;den=elig=low=attempts=0
            for t in range(cfg.steps):
                state=torch.as_tensor(np.stack([obs[i] for i in ids]),dtype=torch.float32)
                logits,_=model(state)
                acts={i:int(x) for i,x in zip(ids,logits.argmax(dim=1))}
                obs,_,done,infos=env.step(acts)
                for i in ids:
                    v=infos[i];b+=float(v['delivered_bits']);en+=float(v['energy_j']);den+=int(v['blocked']);elig+=int(v['blocked']+v['scheduled']);low+=int(v['tx_mode']=='low' and v['scheduled']);attempts+=int(v['scheduled'])
                if done:break
            rows.append(dict(method=method,seed=seed,episode=episode+1,throughput_mbps=b/(len(ids)*cfg.steps*env.slot_s*1e6),energy_mj_step=1000*en/(len(ids)*cfg.steps),denied=den,eligible=elig,low_fraction=low/max(attempts,1)))
    return rows


def write(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summary(rows,limit,denlim):
    out=[]
    for method in sorted(set(x['method'] for x in rows)):
        for seed in sorted(set(x['seed'] for x in rows)):
            r=[x for x in rows if x['method']==method and x['seed']==seed]
            deny=sum(x['denied'] for x in r)/max(sum(x['eligible'] for x in r),1)
            energy=float(np.mean([x['energy_mj_step'] for x in r]));th=float(np.mean([x['throughput_mbps'] for x in r])); low=float(np.mean([x['low_fraction'] for x in r]))
            out.append(dict(method=method,seed=seed,throughput_mbps=th,energy_mj_step=energy,denial=deny,low_fraction=low,joint_feasible=int(deny<=denlim+1e-12 and energy<=limit+1e-12)))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',default='101,102,103');p.add_argument('--calibration-seeds',default='7101,7102,7103,7104,7105');p.add_argument('--episodes',type=int,default=16);p.add_argument('--eval-episodes',type=int,default=8);p.add_argument('--steps',type=int,default=50);p.add_argument('--out',default='v6_pilot');p.add_argument('--energy-cap-mj',type=float,default=None);p.add_argument('--threads',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(a.threads)
    cfg=core.RunConfig(episodes=a.episodes,eval_episodes=a.eval_episodes,steps=a.steps,service_logit_bias=2.,denial_limit=.02)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    calibration=[]
    for seed in map(int,a.calibration_seeds.split(',')):
        for method in ('AlwaysNormal','AlwaysLow','QueueUrgency','Periodic2pct','Mix50','Mix75'):
            calibration.extend(fixed_eval(seed,cfg,method))
    write(out/'calibration.csv',calibration)
    means={m:float(np.mean([r['energy_mj_step'] for r in calibration if r['method']==m])) for m in ['AlwaysNormal','AlwaysLow','QueueUrgency','Periodic2pct','Mix50','Mix75']}
    cap=float(a.energy_cap_mj) if a.energy_cap_mj is not None else (means['AlwaysNormal']+means['AlwaysLow'])/2
    cfg.energy_limit_j_per_step=cap/1000
    print('CALIBRATION',means,'CAP mJ',cap,flush=True)
    allrows=[];hist=[]
    for seed in map(int,a.seeds.split(',')):
        for method in ('AlwaysNormal','AlwaysLow','QueueUrgency','Periodic2pct','Mix50','Mix75'):
            allrows.extend(fixed_eval(seed,cfg,method))
        for constrained in (False,True):
            name='PD-PPO' if constrained else 'PPO'
            print('TRAIN',name,seed,flush=True)
            model,h=core.train(seed,cfg,constrained)
            hist.extend(h)
            allrows.extend(neural_eval(model,seed,cfg,name))
            write(out/'evaluation_episodes.csv',allrows)
            write(out/'per_seed_partial.csv',summary(allrows,cap,cfg.denial_limit))
            print('FINISHED',name,seed,flush=True)
    per=summary(allrows,cap,cfg.denial_limit)
    write(out/'evaluation_episodes.csv',allrows);write(out/'per_seed.csv',per);write(out/'training_diagnostics.csv',hist)
    report=[]
    for method in sorted(set(x['method'] for x in per)):
        r=[x for x in per if x['method']==method]
        report.append(dict(method=method,n=len(r),mean_throughput_mbps=float(np.mean([x['throughput_mbps'] for x in r])),mean_energy_mj_step=float(np.mean([x['energy_mj_step'] for x in r])),mean_denial=float(np.mean([x['denial'] for x in r])),joint_feasible_seeds=sum(x['joint_feasible'] for x in r),low_fraction=float(np.mean([x['low_fraction'] for x in r]))))
    write(out/'summary.csv',report)
    (out/'manifest.json').write_text(json.dumps(dict(status='exploratory pilot',seeds=a.seeds,calibration_seeds=a.calibration_seeds,train_episodes=a.episodes,eval_episodes=a.eval_episodes,calibration_means_mj=means,energy_cap_mj=cap,denial_limit=cfg.denial_limit,power_backoff_db=6.0,comparison='joint feasibility and throughput relative to AlwaysNormal, AlwaysLow, Periodic2pct, QueueUrgency and PPO',limitations='no independent second test campaign; stochastic PHY model synthetic'),indent=2))
    print('SUMMARY',json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
