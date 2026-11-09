"""v8 causal contextual power scheduler; offline fitted and calibrated on disjoint seeds.

This is a contextual regression scheduler, NOT a certified/guaranteed CMDP method.
No denial is issued; energy budget feasibility is tested statistically on unseen seeds.
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import torch
import energy_cmdp as core
from three_action_v7 import env_for, baseline_eval
from three_action_v6 import write, summary

# Avoid using any future channel fade or service outcome in decisions.
def features(unit):
    i=int(unit._pending_device)
    sinr=float(unit.mean_sinr_db[i]+unit.shadow_db_ep[i])
    interf=float(unit.interf_w_ep/max(unit.interf_w_base,1.e-15))
    power=float(10**((unit.tx_power_dbm[i]-30)/10))
    backlog=float(unit.queue_bits[i]/max(unit.bits_per_sample[i],1))
    return np.array([1., np.clip(sinr/20,-2,3),np.clip(sinr/20,-2,3)**2,
        np.log1p(power*1000)/3, np.log1p(backlog)/4,
        np.clip(unit.queue_age[i]/unit.steps_per_ep,0,2),
        np.clip(np.log(max(interf,1e-5)), -3,3)/3,
        np.clip(unit.bits_per_sample[i]/20000,0,5),
        np.clip(unit.energy_per_bit[i]*1e8,0,10)/3],float)

class Regressor:
    def __init__(self,coefs,scales): self.coefs=np.asarray(coefs);self.scales=np.asarray(scales)
    def predict(self,x,action):
        z=np.asarray(x)/self.scales
        b=float(z@self.coefs[action-1,:,0]); e=float(z@self.coefs[action-1,:,1]);
        return max(0.,b),max(0.,e)
    def save(self):return {'coefs':self.coefs.tolist(),'scales':self.scales.tolist()}

def fit(seed_list,cfg,episodes=7,ridge=2.):
    X={1:[],2:[]};Y={1:[],2:[]}
    for seed in seed_list:
        env=env_for(4000000+seed,cfg);ids=list(env.agents)
        for ep in range(episodes):
            env.reset()
            for t in range(cfg.steps):
                act={}; feat={}
                for i,u in env.agents.items():
                    act[i]=1 if (t+i+ep+seed)%2 else 2
                    feat[i]=features(u)
                _,_,done,infos=env.step(act)
                for i in ids:
                    a=act[i];X[a].append(feat[i]);Y[a].append([infos[i]['delivered_bits']/10000,infos[i]['energy_j']*1000])
                if done:break
    stacked=np.vstack(X[1]+X[2]); scales=np.maximum(np.std(stacked,axis=0),.1);scales[0]=1.
    coefs=[]
    for a in (1,2):
        x=np.vstack(X[a])/scales;y=np.asarray(Y[a]);reg=ridge*np.diag([0]+[1]*(x.shape[1]-1))
        coefs.append(np.linalg.solve(x.T@x+reg,x.T@y))
    return Regressor(coefs,scales),{'rows_normal':len(X[1]),'rows_low':len(X[2])}

class Controller:
    def __init__(self,model,price,cap_mj,steps,reserve=0.):
        self.model=model;self.price=float(price);self.cap=float(cap_mj);self.steps=steps;self.reserve=float(reserve)
    def action(self,u):
        x=features(u)
        b1,e1=self.model.predict(x,1);b2,e2=self.model.predict(x,2)
        # Value of normal compared to low, with fixed dual price and budget pacing.
        # No deny action; constraints remain empirically assessed, not guaranteed.
        remaining=max(u.steps_per_ep-u.t,1)
        budget_left=self.cap*self.steps-u.energy_j_ep*1000
        paced=budget_left/remaining
        effective_price=self.price*(1.+2.*np.clip((e2+self.reserve-paced)/max(self.cap,.01),-0.75,2.))
        if paced<e2+self.reserve: return 2
        return 1 if b1-b2>effective_price*max(e1-e2,1e-5) else 2

def evaluate_controller(seed,cfg,method,model=None,price=0.,reserve=0.):
    env=env_for(1_000_000+seed,cfg);ids=list(env.agents);rows=[]
    ctl=Controller(model,price,cfg.energy_limit_j_per_step*1000,cfg.steps,reserve) if model is not None else None
    for ep in range(cfg.eval_episodes):
        env.reset();bits=energy=0.;denied=eligible=low=tx=0
        for t in range(cfg.steps):
            actions={}
            for i,u in env.agents.items():
                if ctl: a=ctl.action(u)
                elif method=='ChannelThreshold':a=1 if u.mean_sinr_db[u._pending_device]+u.shadow_db_ep[u._pending_device]>=9 else 2
                elif method=='AlwaysLow':a=2
                elif method=='AlwaysNormal':a=1
                elif method=='Mix50':a=2 if t%2==0 else 1
                elif method=='BudgetAware':
                    from three_action_v7 import energy_aware_action
                    a=energy_aware_action(u,cfg,cfg.steps-t)
                else:raise ValueError(method)
                actions[i]=a
            _,_,done,infos=env.step(actions)
            for i in ids:
                v=infos[i];bits+=float(v['delivered_bits']);energy+=float(v['energy_j']);denied+=int(v['blocked']);eligible+=int(v['blocked']+v['scheduled']);low+=int(v['tx_mode']=='low' and v['scheduled']);tx+=int(v['scheduled'])
            if done:break
        rows.append(dict(method=method,seed=seed,episode=ep+1,throughput_mbps=bits/(len(ids)*cfg.steps*env.slot_s*1e6),energy_mj_step=1000*energy/(len(ids)*cfg.steps),denied=denied,eligible=eligible,low_fraction=low/max(tx,1)))
    return rows

def choose_price(model,cfg,tune_seeds,prices,reserves):
    trials=[]
    for reserve in reserves:
        for price in prices:
            r=[]
            for seed in tune_seeds:r+=evaluate_controller(seed,cfg,'FittedPacer',model,price,reserve)
            per=summary(r,cfg.energy_limit_j_per_step*1000,cfg.denial_limit)
            trials.append(dict(price=price,reserve=reserve,mean_throughput=float(np.mean([v['throughput_mbps'] for v in per])),
                max_energy=float(max(v['energy_mj_step'] for v in per)),feasible=sum(v['joint_feasible'] for v in per),n=len(per)))
    eligible=[v for v in trials if v['feasible']==v['n']]
    if not eligible:
        # Safe calibration fallback; do not select an infeasible learned controller.
        return None,trials
    return max(eligible,key=lambda x:x['mean_throughput']),trials

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fit-seeds',default='8101,8102,8103,8104,8105,8106')
    p.add_argument('--tune-seeds',default='8201,8202,8203,8204,8205')
    p.add_argument('--test-seeds',default='401,402,403,404,405')
    p.add_argument('--calibration-episodes',type=int,default=6)
    p.add_argument('--eval-episodes',type=int,default=10)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--cap-mj',type=float,default=.2944864377500498)
    p.add_argument('--out',default='v8_pilot')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--test-methods',default='AlwaysNormal,AlwaysLow,Mix50,ChannelThreshold,BudgetAware')
    a=p.parse_args();t0=time.time()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    parse=lambda x:[int(t) for t in x.split(',')]
    fitseeds=parse(a.fit_seeds);tuneseeds=parse(a.tune_seeds);testseeds=parse(a.test_seeds)
    if len(set(fitseeds+tuneseeds+testseeds))!=len(fitseeds+tuneseeds+testseeds):raise ValueError('Seed overlap detected')
    cfg=core.RunConfig(steps=a.steps,eval_episodes=a.eval_episodes,energy_limit_j_per_step=a.cap_mj/1000,denial_limit=.02)
    if a.resume and (out/'model.json').exists() and (out/'calibration_grid.csv').exists():
        print('RESUME frozen calibration',flush=True)
        m=json.loads((out/'model.json').read_text());model=Regressor(m['coefs'],m['scales']);details={'restored':True}
        with (out/'calibration_grid.csv').open(newline='') as fh:
            trials=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(fh)]
        feasible=[v for v in trials if v['feasible']==v['n']]
        chosen=max(feasible,key=lambda x:x['mean_throughput']) if feasible else None
    else:
        print('FITTING',len(fitseeds),'seeds',flush=True)
        model,details=fit(fitseeds,cfg,a.calibration_episodes)
        print('TUNING',len(tuneseeds),'seeds',flush=True)
        chosen,trials=choose_price(model,cfg,tuneseeds,[0,.2,.5,1,2,4,8,16,32],[0,.003,.008])
        write(out/'calibration_grid.csv',trials)
        (out/'model.json').write_text(json.dumps(model.save(),indent=2))
    if chosen is None:
        print('No feasible fitted controller on all tuning seeds; using AlwaysLow as predefined feasible fallback',flush=True)
    else:print('LOCKED CALIBRATION',chosen,flush=True)
    results=[]
    if a.resume and (out/'evaluation_episodes.csv').exists():
        with (out/'evaluation_episodes.csv').open(newline='') as fh:
            for r in csv.DictReader(fh):
                results.append({k:(int(v) if k in ('seed','episode','denied','eligible') else float(v) if k in ('throughput_mbps','energy_mj_step','low_fraction') else v) for k,v in r.items()})
    for seed in testseeds:
        if any(int(r['seed'])==seed for r in results):
            print('SKIP completed seed',seed,flush=True);continue
        print('TEST SEED',seed,flush=True)
        for method in a.test_methods.split(','):
            results+=evaluate_controller(seed,cfg,method)
        if chosen is not None:
            results+=evaluate_controller(seed,cfg,'FittedPacer',model,chosen['price'],chosen['reserve'])
        else:
            results+=evaluate_controller(seed,cfg,'FittedPacerFallbackLow')
        write(out/'evaluation_episodes.csv',results)
        write(out/'per_seed_partial.csv',summary(results,a.cap_mj,cfg.denial_limit))
    per=summary(results,a.cap_mj,cfg.denial_limit)
    write(out/'per_seed.csv',per)
    agg=[]
    for method in sorted(set(r['method'] for r in per)):
        rr=[r for r in per if r['method']==method]
        agg.append(dict(method=method,n=len(rr),throughput_mbps=float(np.mean([r['throughput_mbps'] for r in rr])),energy_mj_step=float(np.mean([r['energy_mj_step'] for r in rr])),denial=float(np.mean([r['denial'] for r in rr])),feasible=sum(r['joint_feasible'] for r in rr)))
    write(out/'summary.csv',agg)
    (out/'manifest.json').write_text(json.dumps(dict(status='development pilot; not confirmatory',fit_seeds=fitseeds,tune_seeds=tuneseeds,test_seeds=testseeds,fit_details=details,chosen=chosen,cap_mj=a.cap_mj,denial_limit=cfg.denial_limit,eval_episodes=cfg.eval_episodes,steps=cfg.steps,elapsed_sec=round(time.time()-t0,1),limitations='No pathwise energy guarantee; physical model synthetic; deterministic baseline is strong; check per-seed performance'),indent=2))
    print('RESULTS',json.dumps(agg,indent=2),flush=True)
if __name__=='__main__':main()
