#!/usr/bin/env python3
"""
Agent-Driven Stock Simulator (Live Plot Edition)

Controls (focus the plot window):
  Space = pause/resume
  +     = faster
  -     = slower
  q     = quit

Run: python agent_market_sim_live.py
"""

import math
import itertools
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#utils
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def ewma(prev, x, alpha):
    return alpha * prev + (1 - alpha) * x

#base agent class
class Agent:
    def __init__(self, name: str, freq: float = 1.0, seed: Optional[int] = None):
        self.name = name
        self.freq = freq
        self.rng = np.random.default_rng(seed)
    def decide(self, t: int, price: np.ndarray, state: dict) -> float:
        return 0.0

#agents
class NoiseTrader(Agent):
    def __init__(self, name="Noise", freq=0.7, size=1.0, bias=0.0, seed=None):
        super().__init__(name, freq, seed); self.size=size; self.bias=bias
    def decide(self, t, price, state):
        if self.rng.random() > self.freq: return 0.0
        sign = 1 if self.rng.random() < 0.5 + 0.5*self.bias else -1
        return sign * self.size * (0.5 + self.rng.random())

class DCAInvestor(Agent):
    def __init__(self, name="DCA", period=20, cash=20.0, seed=None):
        super().__init__(name, 1.0, seed); self.period=max(1,int(period)); self.cash=cash
    def decide(self, t, price, state):
        if t % self.period != 0: return 0.0
        return self.cash / max(price[t], 1e-6)

class MomentumFollower(Agent):
    def __init__(self, name="Momentum", lookback=30, k=1.0, freq=0.8, seed=None):
        super().__init__(name, freq, seed); self.L=max(2,int(lookback)); self.k=k
    def decide(self, t, price, state):
        if self.rng.random()>self.freq or t<self.L: return 0.0
        w=price[t-self.L+1:t+1]; slope=np.polyfit(np.arange(self.L), w, 1)[0]
        return self.k * np.sign(slope) * abs(slope)

class MeanReverter(Agent):
    def __init__(self, name="MeanRevert", L=60, thresh=1.0, k=1.0, freq=0.7, seed=None):
        super().__init__(name, freq, seed); self.L=max(5,int(L)); self.thresh=float(thresh); self.k=float(k)
    def decide(self, t, price, state):
        if self.rng.random()>self.freq or t<self.L: return 0.0
        w=price[t-self.L+1:t+1]; mu=w.mean(); sig=w.std()+1e-9
        z=(price[t]-mu)/sig
        return -self.k*z if abs(z)>self.thresh else 0.0

class ValueInvestor(Agent):
    def __init__(self, name="Value", k=0.5, tol=0.01, freq=0.3, seed=None):
        super().__init__(name, freq, seed); self.k=k; self.tol=tol
    def decide(self, t, price, state):
        if self.rng.random()>self.freq: return 0.0
        V=state['value'][t]; p=price[t]; gap=(V-p)/max(p,1e-6)
        return self.k*gap if abs(gap)>=self.tol else 0.0

class Whale(Agent):
    def __init__(self, name="Whale", p=0.004, mu=0.0, sigma=2.0, impact_boost=2.0, seed=None):
        super().__init__(name, 1.0, seed); self.p=p; self.mu=mu; self.sigma=sigma; self.impact_boost=impact_boost
    def decide(self, t, price, state):
        if self.rng.random()>=self.p: return 0.0
        size=math.exp(self.rng.normal(self.mu,self.sigma)); sign=1 if self.rng.random()<0.5 else -1
        state['whale_hit']=self.impact_boost
        return sign*size



@dataclass
class Config:
    T:int=5000
    start_price:float=100.0
    kappa:float=0.02
    L0:float=120.0
    vol_alpha:float=0.97
    value_drift:float=0.00002
    value_revert:float=0.001
    seed=None

class Simulator:
    def __init__(self, cfg: Config, agents: Dict[str, Tuple[Agent,float]]):
        self.cfg=cfg; self.agents=agents; self.rng=np.random.default_rng(cfg.seed)
        T=cfg.T
        self.price=np.zeros(T); self.value=np.zeros(T); self.vol=np.zeros(T); self.liq=np.zeros(T)
        self.price[0]=cfg.start_price; self.value[0]=cfg.start_price; self.vol[0]=0.20; self.liq[0]=cfg.L0
        self.t=1
    def step(self):
        t=self.t
        if t>=self.cfg.T: return False
        state=dict(price=self.price, value=self.value, vol=self.vol[t-1], liq=self.liq[t-1], whale_hit=0.0)
        total_Q=0.0
        for _,(agent,w) in self.agents.items():
            total_Q += w * agent.decide(t-1, self.price, state)
        kappa=self.cfg.kappa*max(1.0,state['whale_hit'])
        impact = kappa*(total_Q/max(self.liq[t-1],1e-6))
        micro = self.rng.normal(0,0.0015)
        dM = impact + micro
        self.price[t]=max(0.01, self.price[t-1]*(1.0+dM))
        ret=math.log(self.price[t]/self.price[t-1])
        inst=abs(ret)*math.sqrt(252.0)
        self.vol[t]=ewma(self.vol[t-1], inst, self.cfg.vol_alpha)
        self.liq[t]=self.cfg.L0/(1.0+10.0*self.vol[t])
        drift=self.cfg.value_drift+0.001*self.rng.normal()
        long_mean=ewma(self.value[t-1], self.price[t-1], 0.995)
        self.value[t]=self.value[t-1]+drift+self.cfg.value_revert*(long_mean-self.value[t-1])
        self.t += 1
        return True

class LiveRunner:
    def __init__(self, sim: Simulator, ticks_per_frame:int=10):
        self.sim=sim; self.tpf=max(1,int(ticks_per_frame)); self.paused=False
        self.fig,self.ax=plt.subplots(figsize=(11,5))
        self.line_price,=self.ax.plot([],[],label="Price")
        self.line_value,=self.ax.plot([],[],alpha=0.5,label="Value")
        self.ax.set_title("Agent-Driven Stock (LIVE)")
        self.ax.set_xlabel("Tick"); self.ax.set_ylabel("Price"); self.ax.grid(True); self.ax.legend(loc="best")
        self.ax.set_xlim(0, sim.cfg.T); self.ax.set_ylim(0.5*sim.cfg.start_price, 2.5*sim.cfg.start_price)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.ani=None  # hold a strong ref

    def on_key(self, event):
        if event.key == ' ':
            self.paused = not self.paused
        elif event.key == '+':
            self.tpf = min(self.tpf*2, 2000)
        elif event.key == '-':
            self.tpf = max(self.tpf//2, 1)
        elif event.key == 'q':
            plt.close(self.fig)

    def step_frame(self):
        for _ in range(self.tpf):
            if not self.sim.step(): return False
        t=self.sim.t
        x=np.arange(t)
        self.line_price.set_data(x, self.sim.price[:t])
        self.line_value.set_data(x, self.sim.value[:t])
        y=self.sim.price[max(0,t-800):t]
        lo=max(0.01, y.min()*0.97); hi=y.max()*1.03
        self.ax.set_ylim(lo, hi)
        return True

    def _update(self, _frame):
        if not self.paused:
            alive=self.step_frame()
            if not alive and self.ani is not None:
                self.ani.event_source.stop()
        return self.line_price, self.line_value

    def run(self):
        self.ani = FuncAnimation(
            self.fig, self._update,
            frames=itertools.count(), interval=30,
            blit=False, cache_frame_data=False
        )
        plt.tight_layout()
        plt.show()

def main():
    cfg=Config(T=5000, kappa=0.022, L0=140.0)
    agents={
        "Noise":(NoiseTrader(size=1.2,freq=0.7),0.30),
        "DCA":(DCAInvestor(period=28,cash=30.0),0.10),
        "Momentum":(MomentumFollower(lookback=40,k=1.2,freq=0.8),0.22),
        "MeanRevert":(MeanReverter(L=80,thresh=1.0,k=1.0,freq=0.7),0.18),
        "Value":(ValueInvestor(k=2.0,tol=0.008,freq=0.4),0.10),
        "Whale":(Whale(p=0.003,sigma=1.8,impact_boost=2.0),0.10),
    }
    sim=Simulator(cfg,agents)
    LiveRunner(sim, ticks_per_frame=10).run()

if __name__=="__main__":
    main()
