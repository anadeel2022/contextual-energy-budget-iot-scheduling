"""Frozen v8 vs budget-aware channel comparator; evaluation only, no tuning."""
import csv, json, argparse
from pathlib import Path
import numpy as np
from scipy.stats import t
from energy_cmdp import RunConfig
from feasible_v8 import Regressor, Controller, features, env_for
from three_action_v6 import write, summary

CAP=0.2944864377500498

class BudgetThreshold:
    """Causal channel threshold with same fitted energy-cost forecasts and episode pacing.

    Only the normal/low mode selection is modified; the predeclared 9dB cutoff
    is the existing ChannelThreshold benchmark, never optimized on test data.
    """
    def __init__(self,model,cap_mj,steps,threshold_db=9.):
        self.model=model;self.cap=cap_mj;self.steps=steps;self.threshold=threshold_db
    def action(self,u):
        idx=int(u._pending_device)
        estimate=float(u.mean_sinr_db[idx]+u.shadow_db_ep[idx])
        if estimate<self.threshold:return 2
        _,pred_normal=self.model.predict(features(u),1)
        remaining=max(u.steps_per_ep-u.t,1)
        allowance=(self.cap*self.steps-u.energy_j_ep*1000)/remaining
        return 1 if pred_normal<=allowance else 2

def eval_method(seed,cfg,method,model,locked):
    env=env_for(1000000+seed,cfg);ids=list(env.agents)
    fitting=Controller(model,locked['price'],CAP,cfg.steps,locked['reserve'])
    budgeted=BudgetThreshold(model,CAP,cfg.steps)
    rows=[]
    for episode in range(cfg.eval_episodes):
        env.reset();bits=energy=0.;denied=eligible=low=tx=0
        for step in range(cfg.steps):
            actions={}
            for i,u in env.agents.items():
                if method=='FittedPacer': action=fitting.action(u)
                elif method=='BudgetThreshold': action=budgeted.action(u)
                elif method=='ChannelThreshold':
                    ix=int(u._pending_device);action=1 if float(u.mean_sinr_db[ix]+u.shadow_db_ep[ix])>=9 else 2
                else:raise ValueError(method)
                actions[i]=action
            _,_,done,infos=env.step(actions)
            for i in ids:
                v=infos[i];bits+=float(v['delivered_bits']);energy+=float(v['energy_j']);denied+=int(v['blocked']);eligible+=int(v['blocked']+v['scheduled']);low+=int(v['tx_mode']=='low' and v['scheduled']);tx+=int(v['scheduled'])
            if done:break
        rows.append(dict(method=method,seed=seed,episode=episode+1,throughput_mbps=bits/(len(ids)*cfg.steps*env.slot_s*1e6),energy_mj_step=1000*energy/(len(ids)*cfg.steps),denied=denied,eligible=eligible,low_fraction=low/max(tx,1)))
    return rows

def ci(v):
    arr=np.asarray(v,dtype=float);n=len(arr);mean=float(arr.mean());half=float(t.ppf(.975,n-1)*arr.std(ddof=1)/np.sqrt(n)) if n>1 else None
    return dict(n=n,mean=mean,ci95=[mean-half,mean+half] if half is not None else None)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',default='v8_pilot/model.json');p.add_argument('--grid',default='v8_pilot/calibration_grid.csv');p.add_argument('--seeds',default=','.join(map(str,range(501,521))));p.add_argument('--episodes',type=int,default=5);p.add_argument('--out',default='v9_baseline_audit');p.add_argument('--resume',action='store_true');a=p.parse_args()
    output=Path(a.out);output.mkdir(parents=True,exist_ok=True)
    mdl=json.loads(Path(a.model).read_text());model=Regressor(mdl['coefs'],mdl['scales'])
    with open(a.grid,newline='') as f:grid=list(csv.DictReader(f))
    feasible=[r for r in grid if int(float(r['feasible']))==int(float(r['n']))]
    if not feasible:raise ValueError('No feasible locked candidate')
    best=max(feasible,key=lambda r:float(r['mean_throughput']));locked={'price':float(best['price']),'reserve':float(best['reserve'])}
    seeds=[int(s) for s in a.seeds.split(',')]
    if len(seeds)!=len(set(seeds)) or set(seeds)&(set(range(401,406))|set(range(8101,8107))|set(range(8201,8206))):raise ValueError('overlapping or duplicate seeds')
    cfg=RunConfig(steps=50,eval_episodes=a.episodes,energy_limit_j_per_step=CAP/1000,denial_limit=.02)
    path=output/'evaluation_episodes.csv';rows=[]
    if a.resume and path.exists():
        with path.open(newline='') as f:
            rows=[{k:int(v) if k in ('seed','episode','denied','eligible') else float(v) if k in ('throughput_mbps','energy_mj_step','low_fraction') else v for k,v in r.items()} for r in csv.DictReader(f)]
    methods=['FittedPacer','BudgetThreshold','ChannelThreshold']
    for seed in seeds:
        for method in methods:
            if any(int(r['seed'])==seed and r['method']==method for r in rows):continue
            print('AUDIT',seed,method,flush=True)
            rows.extend(eval_method(seed,cfg,method,model,locked));write(path,rows)
    per=summary(rows,CAP,.02);write(output/'per_seed.csv',per)
    stats={};data={}
    for method in methods:
        r=[x for x in per if x['method']==method];er=[x for x in rows if x['method']==method]
        byseed={int(x['seed']):x for x in r};data[method]=byseed
        stats[method]=dict(throughput_mbps=ci([byseed[s]['throughput_mbps'] for s in seeds]),energy_mj_step=ci([byseed[s]['energy_mj_step'] for s in seeds]),seed_feasible=sum(byseed[s]['joint_feasible'] for s in seeds),episode_energy_violations=sum(x['energy_mj_step']>CAP+1e-12 for x in er),episode_denial_violations=sum(x['denied']/max(x['eligible'],1)>.02+1e-12 for x in er),episodes=len(er),mean_denial=sum(x['denied'] for x in er)/max(sum(x['eligible'] for x in er),1))
    diffs={}
    for reference in ('BudgetThreshold','ChannelThreshold'):
        vals=[data['FittedPacer'][s]['throughput_mbps']-data[reference][s]['throughput_mbps'] for s in seeds]
        diffs[reference]=ci(vals)
    report=dict(experiment='Frozen v8 baseline and feasibility audit',methods=stats,paired_gain_fitted_minus=diffs,cap_mj_step=CAP,denial_limit=.02,locked=locked,seeds=seeds,episodes=a.episodes,interpretation='seed-level feasible is not episode-level guarantee; energy cap is a time-average empirical limit; baseline uses same frozen fitted energy model and 9dB channel threshold; no v8 retuning')
    (output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
