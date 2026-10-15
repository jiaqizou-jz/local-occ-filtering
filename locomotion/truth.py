from __future__ import annotations

import os
import numpy as np

os.environ.setdefault('MUJOCO_GL', 'disable')

from gymnasium.envs.mujoco.hopper_v5 import HopperEnv

# ground truth for experiment 3: the same (s, a) stepped in the source and the target simulator
# gymnasium's hopper.xml is the gym / odrl hopper converted to local coordinates (identical body, RK4, dt 0.002 x 4),
# and the odrl gravity variants only add <option gravity=...>, so the shift is applied by scaling model.opt.gravity
# observations are qpos[1:] + clip(qvel, -10, 10); rootx is unobserved and the dynamics are invariant to it

GRAVITY = -9.81

class HopperSim:
    def __init__(self, gravity_scale = 1.):
        self.env = HopperEnv()
        self.env.reset(seed = 0)
        self.env.model.opt.gravity[:] = (0., 0., GRAVITY * gravity_scale)
        self.nq = self.env.model.nq

    def step(self, obs, action):
        qpos = np.concatenate(([0.], obs[:self.nq - 1]))
        qvel = obs[self.nq - 1:]
        self.env.set_state(qpos, qvel)
        return self.env.step(action)[0]

    def step_batch(self, obs, action):
        return np.stack([self.step(o, a) for o, a in zip(obs, action)])

def next_state_pair(obs, action, *, gravity_source = 1., gravity_target):
    src, tar = HopperSim(gravity_source), HopperSim(gravity_target)
    return src.step_batch(obs, action), tar.step_batch(obs, action)

def true_discrepancy(obs, action, *, gravity_target, next_state_std, gravity_source = 1.):
    # normalized l2 distance between the two simulators' next states, 0 iff the dynamics agree at (s, a)
    ns_src, ns_tar = next_state_pair(obs, action, gravity_source = gravity_source, gravity_target = gravity_target)
    return np.linalg.norm((ns_tar - ns_src) / next_state_std, axis = -1), ns_src, ns_tar
