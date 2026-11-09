import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'code'))
import numpy as np
from energy_cmdp import RunConfig
from feasible_v8 import env_for
from env import SpectrumEnv

cfg=RunConfig(steps=50,eval_episodes=2)
env=env_for(1234567,cfg)
unit=next(iter(env.agents.values()))
unit.reset()
assert unit.t==0
# Force multiple arrivals into one finite buffer and ensure drops are counted.
unit.queue_bits[:]=0
unit._pkt_arrivals=[__import__('collections').deque() for _ in range(unit.n_devices)]
unit.sampling_hz[:]=1e8
for _ in range(30): unit._arrivals_step()
assert unit.blocked_arrivals_ep>0
for i in range(unit.n_devices):
 assert abs(unit.queue_bits[i]-sum(p[1] for p in unit._pkt_arrivals[i]))<1e-6
 assert unit.queue_bits[i]<=unit.queue_cap_bits+1e-6
# Exercise ordinary closed-loop steps and verify packet queue invariants continuously.
for seed in (12001,12002,12003):
 world=env_for(seed,cfg)
 for ep in range(3):
  world.reset()
  for t in range(cfg.steps):
   acts={k:(1 if t%2 else 2) for k in world.agents}
   world.step(acts)
   for u in world.agents.values():
    for i in range(u.n_devices):
     assert abs(u.queue_bits[i]-sum(p[1] for p in u._pkt_arrivals[i]))<1e-6
     assert 0<=u.queue_bits[i]<=u.queue_cap_bits+1e-6
     assert u.queue_age[i]>=0
print('PASS packet ledger, overflow and multi-episode invariants')
