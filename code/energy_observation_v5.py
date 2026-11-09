"""Independent calibration and observation-augmented primal-dual PPO evaluation.

Run from MARL-IEEE-ITS. Calibration uses held-out-from-training seed block
7101..7105. Evaluation uses 101..105 mapped to 1_000_000+seed. No tuning
on evaluation data. This is a PILOT, not publication evidence.
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import torch
import energy_cmdp as core

ORIGINAL_ENV_FOR=core.env_for

def augmented_env(seed,cfg):
    e=ORIGINAL_ENV_FOR(seed,cfg)
    for unit in e.agents.values():
        unit.energy_features_enabled=True
        unit.energy_obs_budget_j=cfg.energy_limit_j_per_step or .000343
    return e

core.env_for=augmented_env

def fixed_policy(seed,cfg,method,energy_cap):
    env=augmented_env(1_000_000+seed,cfg); ids=list(env.agents); rows=[]
    for ep in range(cfg.eval_episodes):
        obs=env.reset();b=energy=0.;eligible=denied=0
        for t in range(cfg.steps):
            actions={}
            for aid,u in env.agents.items():
                if method=='AlwaysServe':
                    a=1
                elif method=='Periodic2pct':
                    a=0 if (t+1)%50==0 else 1
                elif method=='NominalCostGate':
                    # A causal non-learning comparator, with at most one denial
                    # per 50-step episode, based on *previous* queue state.
                    # No current-step channel or arrival is observed.
                    c=obs[aid][2*cfg.devices:3*cfg.devices]
                    backlog=obs[aid][3*cfg.devices:4*cfg.devices]
                    candidate=np.flatnonzero(backlog>0)
                    expensive=(len(candidate)>0 and float(np.max(c[candidate]))>float(np.quantile(c,0.8)))
                    a=0 if (expensive and u.denies_this_ep==0 and t>=cfg.steps-4) else 1
                else:raise ValueError(method)
                actions[aid]=a
            obs,_,done,infos=env.step(actions)
            for i in ids:
                v=infos[i];b+=float(v['delivered_bits']);energy+=float(v['energy_j']);denied+=int(v['blocked']);eligible+=int(v['blocked']+v['scheduled'])
            if done:break
        rate=denied/max(eligible,1);en=energy/(len(ids)*cfg.steps)
        rows.append(dict(seed=seed,method=method,eval_episode=ep+1,throughput_mbps=b/(len(ids)*cfg.steps*env.slot_s*1e6),conditional_denial_rate=rate,denied=denied,eligible=eligible,energy_j_per_agent_step=en,feasible=int(rate<=cfg.denial_limit and en<=energy_cap)))
    return rows

def summarize(allrows,cfg,cap):
    results=[]
    for method in sorted(set(r['method'] for r in allrows)):
        for seed in sorted(set(r['seed'] for r in allrows)):
            subset=[r for r in allrows if r['method']==method and r['seed']==seed]
            if not subset: continue
            den=sum(x['denied'] for x in subset);eligible=sum(x['eligible'] for x in subset)
            row=dict(method=method,seed=seed,throughput_mbps=float(np.mean([x['throughput_mbps'] for x in subset])),conditional_denial_rate=den/max(eligible,1),energy_mj_per_agent_step=float(np.mean([x['energy_j_per_agent_step'] for x in subset]))*1000)
            row['joint_feasible']=int(row['conditional_denial_rate']<=cfg.denial_limit and row['energy_mj_per_agent_step']<=1000*cap)
            results.append(row)
    return results

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',default='101,102,103,104,105');p.add_argument('--episodes',type=int,default=24)
    p.add_argument('--eval-episodes',type=int,default=10);p.add_argument('--steps',type=int,default=50)
    p.add_argument('--out',default='energy_observation_v5_results');p.add_argument('--calibration-seeds',default='7101,7102,7103,7104,7105')
    p.add_argument('--energy-cap-mj',type=float,default=None,help='Optional prespecified cap, not tuned to test seeds')
    a=p.parse_args();torch.set_num_threads(1)
    cfg=core.RunConfig(episodes=a.episodes,steps=a.steps,eval_episodes=a.eval_episodes,mode='max_weight',service_logit_bias=2.0,denial_limit=.02)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    calibration=[]
    for s in map(int,a.calibration_seeds.split(',')):
        for method in ('AlwaysServe','Periodic2pct'):
            calibration+=fixed_policy(s,cfg,method,1.0)
    core.write_csv(out/'calibration.csv',calibration)
    means={m:float(np.mean([r['energy_j_per_agent_step'] for r in calibration if r['method']==m])) for m in ('AlwaysServe','Periodic2pct')}
    # Cap is calibrated without looking at testing data; halfway between the
    # *means*, if strictly ordered. Not claimed feasible in every environment.
    cap=float(a.energy_cap_mj)/1000 if a.energy_cap_mj is not None else .5*(means['AlwaysServe']+means['Periodic2pct'])
    cfg.energy_limit_j_per_step=cap
    print('Calibration (mJ/agent-step)',{k:round(v*1000,6) for k,v in means.items()},'frozen cap',cap*1000,flush=True)
    allrows=[]; histories=[]; seeds=list(map(int,a.seeds.split(',')))
    for seed in seeds:
        for method in ('AlwaysServe','Periodic2pct','NominalCostGate'):
            allrows+=fixed_policy(seed,cfg,method,cap)
        for constrained in (False,True):
            method='PD-PPO' if constrained else 'PPO'
            print('Training',method,'seed',seed,flush=True)
            model,hist=core.train(seed,cfg,constrained)
            histories+=hist
            observations=core.evaluate(model,seed,cfg,method)
            allrows.extend(observations)
            results=summarize(allrows,cfg,cap)
            core.write_csv(out/'per_seed_partial.csv',results)
            print(' ',method,'throughput',round(results[-1]['throughput_mbps'],4),'energy mJ',round(results[-1]['energy_mj_per_agent_step'],6),'denial',round(results[-1]['conditional_denial_rate'],4),flush=True)
    # Unified column set for neural and non-learning baselines.
    fields=sorted(set().union(*(r.keys() for r in allrows)))
    with (out/'evaluation_episodes.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(allrows)
    core.write_csv(out/'training_diagnostics.csv',histories)
    per=summarize(allrows,cfg,cap);core.write_csv(out/'per_seed.csv',per)
    summary=[]
    for method in ('AlwaysServe','Periodic2pct','NominalCostGate','PPO','PD-PPO'):
        r=[x for x in per if x['method']==method]
        summary.append(dict(method=method,n=len(r),throughput_mbps=float(np.mean([x['throughput_mbps'] for x in r])),energy_mj_per_agent_step=float(np.mean([x['energy_mj_per_agent_step'] for x in r])),denial_rate=float(np.mean([x['conditional_denial_rate'] for x in r])),joint_feasible_seeds=sum(x['joint_feasible'] for x in r)))
    core.write_csv(out/'summary.csv',summary)
    (out/'manifest.json').write_text(json.dumps({'calibration_seeds':a.calibration_seeds,'test_seeds':seeds,'episodes':a.episodes,'eval_episodes':a.eval_episodes,'steps':a.steps,'calibration_means_j':means,'energy_cap_j':cap,'denial_cap':cfg.denial_limit,'observation_features':'queue,age,per-device nominal tx energy,backlog, prior episode energy, remaining time','evaluation':'independent held-out RNG seeds, deterministic raw network decisions','status':'exploratory; no holdout tuning permitted'},indent=2))
    print('SUMMARY',json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
