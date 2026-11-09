"""Matched PPO and primal-dual PPO, with independent evaluation.

Scientific scope: parallel environments with a shared binary service-admission policy.
Constraints are (a) service-denial fraction conditional on a serviceable queue and
(b) optionally energy consumption per agent-step. This is NOT an arrival-blocking
constraint and does NOT model joint channel allocation.
"""
from __future__ import annotations
import argparse, csv, json, random, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from rl_policy_gradient import ActorCritic, PolicyGradientConfig, _compute_gae_parallel, _policy_update
from sim_runner import make_env

@dataclass
class RunConfig:
    episodes: int = 80
    steps: int = 50
    agents: int = 5
    devices: int = 12
    eval_episodes: int = 30
    denial_limit: float = 0.02
    dual_lr: float = 2.0
    energy_limit_j_per_step: Optional[float] = None
    energy_dual_lr: float = 5.0
    max_dual: float = 20.0
    entropy: float = 0.01
    mode: str = 'max_weight'
    service_logit_bias: float = 0.0

def env_for(seed, cfg):
    return make_env(steps_per_ep=cfg.steps,n_devices=cfg.devices,n_channels=1,seed=seed,
        arrival_scale=1600.,class_mix=(.75,.20,.05),burst_k=4.,device_skew=1.35,
        bits_scale=2.,delay_penalty_w=.012,energy_penalty_w=.0004,
        prio_w=(1.3,1.0,.9),device_select_mode=cfg.mode,n_agents=cfg.agents,
        sinr_sigma_db=4.5,device_jitter_cv=.25)

def validate_cfg(cfg):
    if not (0 <= cfg.denial_limit <= 1): raise ValueError('denial limit must lie in [0,1]')
    if cfg.energy_limit_j_per_step is not None and cfg.energy_limit_j_per_step <= 0: raise ValueError('energy cap must be positive')
    if cfg.dual_lr < 0 or cfg.energy_dual_lr < 0: raise ValueError('dual step sizes must be nonnegative')
    if cfg.episodes < 1 or cfg.eval_episodes < 1 or cfg.steps < 2: raise ValueError('episodes and steps must be positive')

def train(seed, cfg, constrained):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    env=env_for(seed,cfg)
    obs=env.reset();ids=list(obs);model=ActorCritic(np.asarray(obs[ids[0]]).size,getattr(env,'n_actions',2),128)
    if cfg.service_logit_bias:
        with torch.no_grad():
            model.policy_head.bias[1] += cfg.service_logit_bias
    opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    pgcfg=PolicyGradientConfig(algo='ppo', ent_coef=cfg.entropy)
    lam_den=0.;lam_en=0.;history=[]
    for ep in range(cfg.episodes):
        obs=env.reset(); obs_l=[];act_l=[];lp_l=[];val_l=[];rew_l=[];done_l=[]
        den_l=[];elig_l=[];ene_l=[];bits_total=0.
        for t in range(cfg.steps):
            state=torch.from_numpy(np.stack([np.asarray(obs[i],dtype=np.float32) for i in ids]))
            with torch.no_grad():
                a,lp,v=model.act(state)
            actions={i:int(a[j]) for j,i in enumerate(ids)}
            next_obs,rewards,done,infos=env.step(actions)
            denies=np.array([float(infos[i]['blocked']) for i in ids],dtype=np.float32)
            eligible=np.array([float(infos[i]['blocked']+infos[i]['scheduled']) for i in ids],dtype=np.float32)
            energies=np.array([float(infos[i]['energy_j']) for i in ids],dtype=np.float32)
            signal=np.array([float(rewards[i])/1e6 for i in ids],dtype=np.float32)
            if constrained:
                # Additive constraint signal; feasible expected value <=0.
                signal-=lam_den*(denies-cfg.denial_limit*eligible)
                if cfg.energy_limit_j_per_step is not None:
                    signal-=lam_en*(1000.0*(energies-cfg.energy_limit_j_per_step))
            obs_l.append(state.numpy());act_l.append(a.numpy());lp_l.append(lp.numpy());val_l.append(v.numpy());
            rew_l.append(signal);done_l.append(np.full(len(ids),float(done),np.float32))
            den_l.append(denies);elig_l.append(eligible);ene_l.append(energies)
            bits_total+=sum(float(rewards[i]) for i in ids)
            obs=next_obs
            if done:break
        with torch.no_grad():
            tail=torch.from_numpy(np.stack([np.asarray(obs[i],dtype=np.float32) for i in ids]))
            _, nv=model(tail)
        adv,ret=_compute_gae_parallel(np.stack(rew_l),np.stack(val_l),np.stack(done_l),pgcfg.gamma,pgcfg.lam,nv.numpy())
        st=torch.from_numpy(np.concatenate(obs_l));ac=torch.from_numpy(np.concatenate(act_l)).long();
        old=torch.from_numpy(np.concatenate(lp_l));ad=torch.from_numpy(adv.reshape(-1));rt=torch.from_numpy(ret.reshape(-1))
        order=np.arange(len(ac));losses=[]
        for _ in range(pgcfg.update_epochs):
            np.random.shuffle(order)
            for j in range(0,len(order),pgcfg.minibatch_size):
                idx=order[j:j+pgcfg.minibatch_size]
                losses.append(_policy_update(model,opt,st[idx],ac[idx],old[idx],ad[idx],rt[idx],pgcfg))
        den_count=float(np.sum(den_l));elig_count=float(np.sum(elig_l));en_mean=float(np.mean(ene_l))
        den_rate=den_count/max(elig_count,1.)
        if constrained:
            # Unbiased sample estimate for conditional denial constraint.
            if elig_count>0:
                lam_den=float(np.clip(lam_den+cfg.dual_lr*(den_rate-cfg.denial_limit),0,cfg.max_dual))
            if cfg.energy_limit_j_per_step is not None:
                lam_en=float(np.clip(lam_en+cfg.energy_dual_lr*(1000.0*(en_mean-cfg.energy_limit_j_per_step)),0,cfg.max_dual))
        history.append(dict(seed=seed,method='PD-PPO' if constrained else 'PPO',episode=ep+1,
            train_denial_rate=den_rate,train_energy_j_per_step=en_mean,
            train_bits_per_agent_step=bits_total/(len(ids)*len(rew_l)),lambda_denial=lam_den,
            lambda_energy=lam_en,policy_loss=float(np.mean([x[0] for x in losses]))))
    return model,history

@torch.no_grad()
def evaluate(model,seed,cfg,method):
    # Do not reuse training RNG or training environment; both policies evaluated
    # on exactly the same held-out environment seed for each replicate.
    env=env_for(1_000_000+seed,cfg)
    ids=list(env.agents);records=[]
    model.eval()
    for e in range(cfg.eval_episodes):
        obs=env.reset(); denied=eligible=arrivals=0;energy=bits=0.;delays=[]
        for _ in range(cfg.steps):
            state=torch.from_numpy(np.stack([np.asarray(obs[i],dtype=np.float32) for i in ids]))
            logits,_=model(state)
            # Fixed deterministic argmax policy; identical for both methods.
            actions={i:int(a) for i,a in zip(ids,logits.argmax(dim=1).numpy())}
            obs,_,done,infos=env.step(actions)
            for i in ids:
                inf=infos[i]; denied+=int(inf['blocked']);eligible+=int(inf['blocked']+inf['scheduled'])
                arrivals+=int(inf['arrivals']);energy+=float(inf['energy_j']);bits+=float(inf['delivered_bits'])
                if 'delay_ms' in inf:delays.append(float(inf['delay_ms']))
            if done:break
        records.append(dict(seed=seed,method=method,mode=cfg.mode,eval_episode=e+1,
            delivered_bits=bits,throughput_mbps=bits/(len(ids)*cfg.steps*env.slot_s*1e6),
            denied=denied,eligible=eligible,arrivals=arrivals,
            conditional_denial_rate=denied/max(eligible,1),
            legacy_denials_per_arrival=denied/max(arrivals,1),
            energy_j_per_agent_step=energy/(len(ids)*cfg.steps),
            energy_eff_mbj=(bits/max(energy,1e-12))/1e6,
            avg_served_packet_delay_ms=float(np.mean(delays)) if delays else float('nan'),
            feasible_denial=int(eligible>0 and denied/eligible<=cfg.denial_limit),
            feasible_energy=int(cfg.energy_limit_j_per_step is None or energy/(len(ids)*cfg.steps)<=cfg.energy_limit_j_per_step)))
    return records

def write_csv(path,rows):
    with open(path,'w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def aggregate(records, cfg):
    from scipy.stats import t
    rows=[]; seeds=sorted(set(x['seed'] for x in records))
    keys=['throughput_mbps','conditional_denial_rate','energy_j_per_agent_step','energy_eff_mbj','avg_served_packet_delay_ms']
    for method in ('PPO','PD-PPO'):
        for seed in seeds:
            a=[r for r in records if r['seed']==seed and r['method']==method]
            item={'method':method,'seed':seed}
            for k in keys:
                vals=np.array([x[k] for x in a],dtype=float)
                item[k]=float(np.mean(vals[np.isfinite(vals)])) if np.isfinite(vals).any() else float('nan')
            # Pooled ratio is correct for denial probability, rather than episode rate average
            item['conditional_denial_rate']=sum(x['denied'] for x in a)/max(sum(x['eligible'] for x in a),1)
            item['denial_feasible']=int(item['conditional_denial_rate']<=cfg.denial_limit)
            if cfg.energy_limit_j_per_step is not None:item['energy_feasible']=int(item['energy_j_per_agent_step']<=cfg.energy_limit_j_per_step)
            rows.append(item)
    contrasts=[]
    for k in keys:
        d=np.array([next(x[k] for x in rows if x['seed']==s and x['method']=='PD-PPO')-
                    next(x[k] for x in rows if x['seed']==s and x['method']=='PPO') for s in seeds])
        d=d[np.isfinite(d)]
        mean=float(np.mean(d)) if len(d)>0 else float('nan');sd=float(np.std(d,ddof=1)) if len(d)>1 else float('nan')
        half=float(t.ppf(.975,len(d)-1)*sd/np.sqrt(len(d))) if len(d)>1 else float('nan')
        contrasts.append(dict(metric=k,paired_mean_difference=mean,ci95_low=mean-half,ci95_high=mean+half,n_seeds=len(d)))
    return rows,contrasts

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',default='101,102,103');p.add_argument('--episodes',type=int,default=80)
    p.add_argument('--steps',type=int,default=50);p.add_argument('--eval-episodes',type=int,default=30)
    p.add_argument('--agents',type=int,default=5);p.add_argument('--devices',type=int,default=12)
    p.add_argument('--mode',choices=['max_weight','max_age','round_robin'],default='max_weight')
    p.add_argument('--denial-limit',type=float,default=.02);p.add_argument('--dual-lr',type=float,default=2.)
    p.add_argument('--energy-limit',type=float,default=0.000343);p.add_argument('--energy-dual-lr',type=float,default=5.)
    p.add_argument('--out',default='cmdp_results');p.add_argument('--threads',type=int,default=1);p.add_argument('--resume',action='store_true');p.add_argument('--service-bias',type=float,default=2.)
    a=p.parse_args();torch.set_num_threads(a.threads)
    cfg=RunConfig(episodes=a.episodes,steps=a.steps,agents=a.agents,devices=a.devices,eval_episodes=a.eval_episodes,
      denial_limit=a.denial_limit,dual_lr=a.dual_lr,energy_limit_j_per_step=a.energy_limit,energy_dual_lr=a.energy_dual_lr,mode=a.mode,service_logit_bias=a.service_bias)
    validate_cfg(cfg);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    seeds=[int(s) for s in a.seeds.split(',')];start=time.time();results=[];trainlog=[]
    if a.resume and (out/'evaluation_episodes.csv').exists():
        with (out/'evaluation_episodes.csv').open(newline='',encoding='utf8') as f:
            for row in csv.DictReader(f):
                row['seed']=int(row['seed'])
                for k,v in list(row.items()):
                    if k not in ('seed','method','mode'):
                        try:row[k]=float(v)
                        except (ValueError,TypeError):pass
                results.append(row)
        if (out/'training_diagnostics.csv').exists():
            with (out/'training_diagnostics.csv').open(newline='',encoding='utf8') as f:trainlog=list(csv.DictReader(f))
        print(f'RESUME: existing evaluation rows={len(results)}',flush=True)
    for s in seeds:
        for constrained in (False,True):
            name='PD-PPO' if constrained else 'PPO'
            if any(int(x['seed'])==s and x['method']==name for x in results):
                print(f'SKIP completed {name} seed={s}',flush=True);continue
            print(f'RUN {name} seed={s} mode={cfg.mode} train={cfg.episodes} eval={cfg.eval_episodes}',flush=True)
            model,log=train(s,cfg,constrained);trainlog.extend(log);results.extend(evaluate(model,s,cfg,name))
            print(f'  final multipliers: denial={log[-1]["lambda_denial"]:.4f} energy={log[-1]["lambda_energy"]:.4f}',flush=True)
            write_csv(out/'evaluation_episodes.csv',results);write_csv(out/'training_diagnostics.csv',trainlog)
    per_seed,contrast=aggregate(results,cfg)
    write_csv(out/'evaluation_episodes.csv',results);write_csv(out/'per_seed.csv',per_seed)
    write_csv(out/'paired_contrasts.csv',contrast);write_csv(out/'training_diagnostics.csv',trainlog)
    (out/'manifest.json').write_text(json.dumps(dict(config=asdict(cfg),seeds=seeds,elapsed_seconds=round(time.time()-start,2),
        evaluation='held-out seeds, deterministic argmax, same seeds for both methods',source='confirm_cmdp.py'),indent=2))
    for r in contrast:print(f'{r["metric"]}: PD-PPO minus PPO {r["paired_mean_difference"]:.6g} CI [{r["ci95_low"]:.6g},{r["ci95_high"]:.6g}]',flush=True)
if __name__=='__main__':main()
