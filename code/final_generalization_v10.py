"""Frozen contextual controller external-condition generalization audit.

No fitting or selection on evaluation seeds. Scenario interventions apply before
any reset, symmetrically to all policies. All episodes and seed ids independent
of v8/v9 fit/calibration/test blocks.
"""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from scipy.stats import t
from energy_cmdp import RunConfig
from feasible_v8 import Regressor,Controller,features,env_for
from three_action_v6 import write,summary
from audit_v9 import BudgetThreshold,CAP

SCENARIOS = {
 'nominal':(1.0,0.0),
 'light_traffic':(.70,0.0),
 'heavy_traffic':(1.30,0.0),
 'weaker_channel':(1.0,-3.0),
 'stronger_channel':(1.0,3.0),
 'heavy_weak':(1.30,-3.0),
}
METHODS=('FittedPacer','ChannelThreshold','BudgetThreshold','CalibratedThreshold9','AggressiveThreshold6','AlwaysLow','AlwaysNormal')

def action(name,u,model,fit):
 if name=='FittedPacer': return fit.action(u)
 if name=='AlwaysLow':return 2
 if name=='AlwaysNormal':return 1
 if name=='ChannelThreshold':return 1 if float(u.mean_sinr_db[u._pending_device]+u.shadow_db_ep[u._pending_device])>=9 else 2
 if name=='BudgetThreshold':return BudgetThreshold(model,CAP,u.steps_per_ep).action(u)
 cutoff,margin = (9.,1.0) if name=='CalibratedThreshold9' else (6.,4.)
 idx=int(u._pending_device)
 if float(u.mean_sinr_db[idx]+u.shadow_db_ep[idx])<cutoff:return 2
 _,pred=model.predict(features(u),1)
 remaining=max(u.steps_per_ep-u.t,1)
 allowance=(CAP*u.steps_per_ep-u.energy_j_ep*1000)/remaining
 return 1 if pred<=allowance*(1+margin) else 2

def evaluate(seed,scenario,cfg,name,model,locked):
 env=env_for(1_000_000+seed,cfg)
 scale,db=SCENARIOS[scenario]
 for unit in env.agents.values():
  unit.sampling_hz = np.asarray(unit.sampling_hz)*scale
  unit.mean_sinr_db = np.asarray(unit.mean_sinr_db)+db
  # Only causally known parameters change; actual physical-layer draws remain stochastic.
 ids=list(env.agents)
 fit=Controller(model,locked['price'],CAP,cfg.steps,locked['reserve'])
 result=[]
 for ep in range(cfg.eval_episodes):
  env.reset();bits=energy=0.;denied=eligible=low=tx=0
  for step in range(cfg.steps):
   actions={i:action(name,u,model,fit) for i,u in env.agents.items()}
   _,_,done,infos=env.step(actions)
   for i in ids:
    v=infos[i];bits+=float(v['delivered_bits']);energy+=float(v['energy_j']);denied+=int(v['blocked']);eligible+=int(v['blocked']+v['scheduled']);low+=int(v['tx_mode']=='low' and v['scheduled']);tx+=int(v['scheduled'])
   if done:break
  result.append(dict(scenario=scenario,method=name,seed=seed,episode=ep+1,throughput_mbps=bits/(len(ids)*cfg.steps*env.slot_s*1e6),energy_mj_step=1000*energy/(len(ids)*cfg.steps),denied=denied,eligible=eligible,low_fraction=low/max(tx,1)))
 return result

def interval(vals):
 a=np.asarray(vals,float);n=len(a);mean=float(a.mean());sd=float(a.std(ddof=1)) if n>1 else None
 half=float(t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else None
 return dict(mean=mean,ci95=[mean-half,mean+half] if half is not None else None,n=n)

def main():
 p=argparse.ArgumentParser();p.add_argument('--seeds',default='1001:1020');p.add_argument('--episodes',type=int,default=5);p.add_argument('--scenarios',default=','.join(SCENARIOS));p.add_argument('--out',default='final_generalization_v10');p.add_argument('--resume',action='store_true');a=p.parse_args()
 seeds=list(range(*[int(x) for x in a.seeds.split(':')])) if ':' in a.seeds else [int(x) for x in a.seeds.split(',')]
 scenarios=a.scenarios.split(',');assert all(s in SCENARIOS for s in scenarios)
 forbidden=set(range(501,521))|set(range(401,406))|set(range(8101,8107))|set(range(8201,8206))|set(range(8301,8306))
 assert not set(seeds)&forbidden and len(set(seeds))==len(seeds)
 path=Path(a.out);path.mkdir(parents=True,exist_ok=True)
 origin=Path(__file__).parent
 d=json.loads((origin/'v8_pilot/model.json').read_text());model=Regressor(d['coefs'],d['scales'])
 with (origin/'v8_pilot/calibration_grid.csv').open(newline='') as f:g=list(csv.DictReader(f))
 feasible=[r for r in g if int(float(r['feasible']))==int(float(r['n']))];best=max(feasible,key=lambda r:float(r['mean_throughput']))
 locked={'price':float(best['price']),'reserve':float(best['reserve'])}
 cfg=RunConfig(steps=50,eval_episodes=a.episodes,energy_limit_j_per_step=CAP/1000,denial_limit=.02)
 saved=path/'episodes.csv';rows=[]
 if a.resume and saved.exists():
  with saved.open(newline='') as f:
   rows=[{k:(int(v) if k in ('seed','episode','denied','eligible') else float(v) if k in ('throughput_mbps','energy_mj_step','low_fraction') else v) for k,v in r.items()} for r in csv.DictReader(f)]
 start=time.monotonic()
 for scenario in scenarios:
  for seed in seeds:
   for method in METHODS:
    if any(r['scenario']==scenario and int(r['seed'])==seed and r['method']==method for r in rows):continue
    rows+=evaluate(seed,scenario,cfg,method,model,locked)
   write(saved,rows)
   print('COMPLETE',scenario,seed,'elapsed',round(time.monotonic()-start,1),'s',flush=True)
 report={'protocol':'frozen v8 regression/controller, fixed budget, frozen v9 baselines; independent scenario-shift seeds; no retuning','seed_ids':seeds,'episodes_per_seed':a.episodes,'steps_per_episode':50,'agents':5,'energy_limit_mj_step':CAP,'denial_limit':.02,'scenarios':{},'paired_differences':{}}
 compact=[]
 for scenario in scenarios:
  selected=[r for r in rows if r['scenario']==scenario]
  report['scenarios'][scenario]={}
  report['paired_differences'][scenario]={}
  v={}
  for method in METHODS:
   group=[r for r in selected if r['method']==method];per=summary(group,CAP,.02)
   byseed={int(r['seed']):r for r in per};v[method]=byseed
   data=dict(throughput=interval([byseed[s]['throughput_mbps'] for s in seeds]),energy=interval([byseed[s]['energy_mj_step'] for s in seeds]),seed_average_feasible=sum(byseed[s]['joint_feasible'] for s in seeds),episode_energy_violations=sum(float(r['energy_mj_step'])>CAP+1e-12 for r in group),episode_denial_violations=sum(int(r['denied'])/max(int(r['eligible']),1)>.02+1e-12 for r in group),episodes=len(group),pooled_denial=sum(int(r['denied']) for r in group)/max(sum(int(r['eligible']) for r in group),1))
   report['scenarios'][scenario][method]=data
   compact.append(dict(scenario=scenario,method=method,throughput_mbps=data['throughput']['mean'],energy_mj_step=data['energy']['mean'],seed_feasible=data['seed_average_feasible'],n_seeds=len(seeds),episode_energy_violations=data['episode_energy_violations'],n_episodes=len(group)))
  for method in METHODS[1:]:
   report['paired_differences'][scenario][method]=interval([v['FittedPacer'][s]['throughput_mbps']-v[method][s]['throughput_mbps'] for s in seeds])
 write(path/'summary.csv',compact)
 (path/'report.json').write_text(json.dumps(report,indent=2))
 print('FINAL SUMMARY',json.dumps(compact,indent=2),flush=True)
if __name__=='__main__':main()
