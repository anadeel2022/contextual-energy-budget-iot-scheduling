"""Frozen V21 corrected FIFO long-run evaluator. No training or tuning on evaluation seeds."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
from feasible_v8 import Regressor, Controller, features, env_for
from audit_v9 import CAP
from energy_cmdp import RunConfig
from final_generalization_v19 import SCENARIOS, action as baseline_action

POLICIES = ('Guarded', 'Fixed9dB', 'Unguarded')
FIELDS = ('scenario', 'method', 'seed', 'episode', 'throughput_mbps', 'energy_mj_step', 'low_fraction', 'denied', 'eligible')
CHECKSUM_FILES = ('code/env.py', 'code/phy_layer.py', 'code/feasible_v8.py', 'code/final_generalization_v19.py', 'code/v19_fitted/model.json', 'development_selection.json')

def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()

def write_csv_atomic(path, rows, fields):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as file:
        wr = csv.DictWriter(file, fieldnames=fields); wr.writeheader(); wr.writerows(rows)
    os.replace(tmp, path)

def read_records(path):
    if not path.exists(): return []
    with path.open(newline='', encoding='utf-8') as file:
        return [{**r, 'seed':int(r['seed']), 'episode':int(r['episode']), **{k:float(r[k]) for k in ('throughput_mbps','energy_mj_step','low_fraction')}, **{k:int(r[k]) for k in ('denied','eligible')}} for r in csv.DictReader(file)]

def run_group(seed, scenario, method, episodes, steps, model):
    cfg = RunConfig(steps=steps, eval_episodes=episodes, energy_limit_j_per_step=CAP/1000., denial_limit=.02)
    env = env_for(1_000_000 + seed, cfg)
    scale, db = SCENARIOS[scenario]
    for unit in env.agents.values():
        unit.sampling_hz = np.asarray(unit.sampling_hz) * scale
        unit.mean_sinr_db = np.asarray(unit.mean_sinr_db) + db
    ctl = Controller(model, 4., CAP, steps, 0.)
    rows=[]
    for episode in range(1,episodes+1):
        env.reset(); bits=energy=0.; eligible=denied=low=0
        for _ in range(steps):
            actions={}
            for i, unit in env.agents.items():
                if method=='Guarded':
                    x=features(unit); _,en=model.predict(x,1); _,el=model.predict(x,2)
                    allowance=(CAP*steps - unit.energy_j_ep*1000.)/max(unit.steps_per_ep-unit.t,1)
                    action=2 if allowance<el or en>allowance else ctl.action(unit)
                elif method=='Unguarded': action=ctl.action(unit)
                else: action=baseline_action('ChannelThreshold',unit,model,ctl)
                actions[i]=action
            _,_,done,info=env.step(actions)
            for v in info.values():
                bits+=float(v['delivered_bits']);energy+=float(v['energy_j'])
                denied+=int(v['blocked']);eligible+=int(v['blocked']+v['scheduled'])
                low+=int(v['tx_mode']=='low' and v['scheduled'])
            if done: break
        rows.append(dict(scenario=scenario,method=method,seed=seed,episode=episode,throughput_mbps=bits/(len(env.agents)*steps*env.slot_s*1e6),energy_mj_step=1000.*energy/(len(env.agents)*steps),low_fraction=low/max(eligible-denied,1),denied=denied,eligible=eligible))
    return rows

def interval(vals):
    vals=np.asarray(vals,dtype=float); n=len(vals); m=float(np.mean(vals))
    h=float(student_t.ppf(.975,n-1)*np.std(vals,ddof=1)/np.sqrt(n)) if n>1 else float('nan')
    return m,m-h,m+h

def analyze(rows, out, episodes, seeds, scenarios):
    summary=[];pairs=[]; per_seed={}
    for scene in scenarios:
        for method in POLICIES:
            subset=[r for r in rows if r['scenario']==scene and r['method']==method]
            d={s:sorted((r for r in subset if r['seed']==s),key=lambda r:r['episode']) for s in seeds}
            assert all(len(v)==episodes and [r['episode'] for r in v]==list(range(1,episodes+1)) for v in d.values()), 'Missing/duplicate episode(s)'
            avg=np.array([np.mean([r['throughput_mbps'] for r in d[s]]) for s in seeds]); energy=np.array([np.mean([r['energy_mj_step'] for r in d[s]]) for s in seeds]); per_seed[scene,method]=avg
            summary.append(dict(scenario=scene,method=method,n_seeds=len(seeds),n_episodes=len(subset),throughput_mbps=float(avg.mean()),energy_mj_step=float(energy.mean()),seed_feasible=int(sum(energy<=CAP+1e-12)),episode_violations=int(sum(r['energy_mj_step']>CAP+1e-12 for r in subset)),low_fraction=float(np.mean([r['low_fraction'] for r in subset])),denial_rate=sum(r['denied'] for r in subset)/max(sum(r['eligible'] for r in subset),1)))
        for target in ('Fixed9dB','Unguarded'):
            diff=per_seed[scene,'Guarded']-per_seed[scene,target];mean,lo,hi=interval(diff)
            pairs.append(dict(scenario=scene,comparison='Guarded_minus_'+target,n_seeds=len(seeds),diff_mbps=mean,ci_low=lo,ci_high=hi))
    write_csv_atomic(out/'summary.csv',summary,list(summary[0]))
    write_csv_atomic(out/'paired_ci.csv',pairs,list(pairs[0]))
    # A seed-cluster is the statistical replication unit, not an individual episode.
    print('\nFINAL COMPARISON (guard minus fixed)',flush=True)
    for p in pairs:
        if p['comparison']=='Guarded_minus_Fixed9dB':print(f" {p['scenario']:<17} {p['diff_mbps']:+.4f} [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}] Mbps",flush=True)
    print(' Total guarded episode violations:',sum(x['episode_violations'] for x in summary if x['method']=='Guarded'),flush=True)
    from make_figures import make_plots
    make_plots(rows, summary, pairs, seeds, scenarios, out, CAP)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds',type=int,default=20,help='Number of previously unused independent seed clusters; e.g. 200')
    ap.add_argument('--start',type=int,default=20001,help='First seed, inclusive; avoid seeds used in development or earlier final reports')
    ap.add_argument('--episodes',type=int,default=5,help='Episodes per seed and scenario')
    ap.add_argument('--steps',type=int,default=50,help='Slots per episode; frozen V21 evaluation uses 50')
    ap.add_argument('--out',default='outputs/final_long',help='Separate output directory for each run')
    ap.add_argument('--scenarios',default=','.join(SCENARIOS),help='Comma separated scenario IDs')
    ap.add_argument('--checkpoint-every',type=int,default=1,help='Save every N completed seed/scenario groups')
    ap.add_argument('--max-groups',type=int,default=0,help='For quick smoke testing; 0 runs all groups')
    args=ap.parse_args()
    if args.seeds<2 or args.episodes<1 or args.steps<1 or args.checkpoint_every<1:ap.error('Invalid counts')
    if args.steps!=50:ap.error('The frozen V21 controller and energy calibration were assessed at 50 slots. Do not change this in confirmatory runs.')
    scenarios=args.scenarios.split(',')
    if len(set(scenarios))!=len(scenarios) or any(s not in SCENARIOS for s in scenarios):ap.error('Invalid or repeated scenarios')
    seeds=list(range(args.start,args.start+args.seeds))
    prohibited=set(range(8101,8107))|set(range(8201,8206))|set(range(8301,8306))|set(range(9201,9206))|set(range(11001,11021))|set(range(13001,13007))|set(range(14001,14021))|set(range(1001,1021))
    if set(seeds)&prohibited:ap.error('Seed range overlaps known fitting, calibration, development, or prior evaluation seeds.')
    out=(ROOT/args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    # Preserve run identity and prevent accidental reuse of checkpoints with different experiment settings.
    design=dict(seed_start=args.start,n_seeds=args.seeds,episodes=args.episodes,steps=args.steps,scenarios=scenarios,methods=list(POLICIES),energy_cap_mj_step=CAP,model_source='v19 corrected FIFO frozen fit',guard='guard_0.0',files_sha256={f:sha(f) for f in CHECKSUM_FILES})
    protocol_path=out/'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())!=design:raise RuntimeError('Output folder already contains a DIFFERENT experiment. Choose a new --out folder.')
    else:protocol_path.write_text(json.dumps(design,indent=2))
    selection=json.loads((ROOT/'development_selection.json').read_text())
    if selection['selected_method']!='guard_0.0':raise RuntimeError('Frozen selection differs from guard_0.0')
    md=json.loads((ROOT/'code/v19_fitted/model.json').read_text());model=Regressor(md['coefs'],md['scales'])
    path=out/'episodes.csv';rows=read_records(path)
    keys=set();groups={}
    for r in rows:
        key=(r['scenario'],r['method'],r['seed'],r['episode'])
        if key in keys:raise RuntimeError('Duplicate record found in checkpoint')
        keys.add(key);groups.setdefault(key[:3],[]).append(r)
        if r['scenario'] not in scenarios or r['method'] not in POLICIES or r['seed'] not in seeds or not 1<=r['episode']<=args.episodes:raise RuntimeError('Checkpoint does not match protocol')
    for key,v in groups.items():
        if len(v)!=args.episodes:raise RuntimeError(f'Incomplete method group in checkpoint: {key}. Restore previous complete checkpoint.')
    started=time.perf_counter();new=0;newgroups=0
    for scene in scenarios:
        for seed in seeds:
            for method in POLICIES:
                if (scene,method,seed) in groups:continue
                group=run_group(seed,scene,method,args.episodes,args.steps,model)
                rows.extend(group);groups[scene,method,seed]=group
                new+=1
            newgroups+=1
            if newgroups%args.checkpoint_every==0:
                write_csv_atomic(path,rows,FIELDS)
            done=sum(len(v)==args.episodes for v in groups.values())
            print(f'PROGRESS {scene:>18} seed={seed} methods={done}/{len(scenarios)*len(seeds)*len(POLICIES)} elapsed_s={time.perf_counter()-started:.1f}',flush=True)
            if args.max_groups and newgroups>=args.max_groups:
                write_csv_atomic(path,rows,FIELDS)
                print('PARTIAL CHECKPOINT saved. Rerun the SAME command without --max-groups to resume.',flush=True)
                return
    write_csv_atomic(path,rows,FIELDS)
    assert len(rows)==len(scenarios)*len(seeds)*len(POLICIES)*args.episodes
    analyze(rows,out,args.episodes,seeds,scenarios)
    (out/'complete.json').write_text(json.dumps({'complete':True,'records':len(rows),'elapsed_s':time.perf_counter()-started},indent=2))
    print('COMPLETE:',out,flush=True)

if __name__=='__main__':main()
