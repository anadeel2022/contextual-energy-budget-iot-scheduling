import sys
from pathlib import Path
from collections import deque
sys.path.insert(0,str(Path(__file__).parent/'code'))
from energy_cmdp import RunConfig
from feasible_v8 import env_for
from phy_layer import PHYLayer
class FixedPHY:
 def normal(self):return 0.
 def random(self):return .99999
cfg=RunConfig(steps=50,eval_episodes=1)
u=next(iter(env_for(932134,cfg).agents.values()))
u.reset();idx=0
u.sampling_hz[:]=0.
u.queue_bits[:]=0.
u.queue_age[:]=0
u._pkt_arrivals=[deque() for _ in range(u.n_devices)]
u._pkt_arrivals[idx]=deque([[0,4000.]])
u.queue_bits[idx]=4000.
u._pending_device=idx;u._pending_arrivals=0;u.phy_rng=FixedPHY()
old=PHYLayer.rate_multiplier_from_sinr_db
try:
 PHYLayer.rate_multiplier_from_sinr_db=staticmethod(lambda x:.5)
 _,_,_,info=u.step(1)
 assert info['delivered_bits']==2000.,info
 assert len(u._pkt_arrivals[idx])==1 and u._pkt_arrivals[idx][0][1]==2000.
 assert u.queue_age[idx]==1
 assert len(u.delay_samples_ms_ep)==0
 u._pending_device=idx;u._pending_arrivals=0
 _,_,_,info=u.step(1)
 assert info['delivered_bits']==2000.
 assert len(u._pkt_arrivals[idx])==0 and u.queue_bits[idx]==0
 assert len(u.delay_samples_ms_ep)==1
 # Several packets can be served from a single opportunity up to the capacity limit.
 u._pkt_arrivals[idx]=deque([[2,4000.],[2,4000.]])
 u.queue_bits[idx]=8000.
 u._pending_device=idx;u._pending_arrivals=0
 PHYLayer.rate_multiplier_from_sinr_db=staticmethod(lambda x:1.45)
 _,_,_,info=u.step(1)
 assert abs(info['delivered_bits']-5800)<1e-8
 assert len(u._pkt_arrivals[idx])==1 and abs(u._pkt_arrivals[idx][0][1]-2200)<1e-8
 assert len(u.delay_samples_ms_ep)==2
finally:
 PHYLayer.rate_multiplier_from_sinr_db=old
print('PASS deterministic partial packet, completion delay, multi-packet transmission, and single-step age')
