from __future__ import annotations

import os
import numpy as np
import mujoco

os.environ.setdefault('MUJOCO_GL', 'disable')

from gymnasium.envs.mujoco.hopper_v5 import HopperEnv

# ground truth for experiment 3: the same (s, a) stepped in the source and the target simulator
# gymnasium's hopper.xml is the gym / odrl hopper converted to local coordinates (identical body, RK4, dt 0.002 x 4),
# and the odrl gravity / friction variants only edit option gravity / geom friction, so both are applied programmatically
# observations are qpos[1:] + clip(qvel, -10, 10); rootx is unobserved and the dynamics are invariant to it

GRAVITY = -9.81
BODY_GEOMS = ('torso_geom', 'thigh_geom', 'leg_geom', 'foot_geom')

# odrl shifts reproduced on gymnasium's model:
#   gravity_<x>:  <option gravity="0 0 -9.81 x">
#   friction_<x>: sliding friction of the four body geoms scaled by x (floor untouched; contact friction is the max of the pair)

class HopperSim:
    def __init__(self, shift = None, level = 1.):
        self.env = HopperEnv()
        self.env.reset(seed = 0)
        model = self.env.model
        if shift == 'gravity':
            model.opt.gravity[:] = (0., 0., GRAVITY * level)
        elif shift == 'friction':
            for name in BODY_GEOMS:
                model.geom_friction[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name), 0] *= level
        elif shift is not None:
            raise NotImplementedError(shift)
        self.nq = model.nq

    def step(self, obs, action):
        qpos = np.concatenate(([0.], obs[:self.nq - 1]))
        qvel = obs[self.nq - 1:]
        self.env.set_state(qpos, qvel)
        return self.env.step(action)[0]

    def step_batch(self, obs, action):
        return np.stack([self.step(o, a) for o, a in zip(obs, action)])

def next_state_pair(obs, action, *, shift, level):
    src, tar = HopperSim(), HopperSim(shift, level)
    return src.step_batch(obs, action), tar.step_batch(obs, action)

def true_discrepancy(obs, action, *, shift, level, next_state_std):
    # normalized l2 distance between the two simulators' next states, 0 iff the dynamics agree at (s, a)
    ns_src, ns_tar = next_state_pair(obs, action, shift = shift, level = level)
    return np.linalg.norm((ns_tar - ns_src) / next_state_std, axis = -1), ns_src, ns_tar
