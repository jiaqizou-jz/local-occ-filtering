from __future__ import annotations

import numpy as np
import gym

from envs.mujoco.call_mujoco_env import call_mujoco_env

# mujoco is deterministic, so the per-(s, a) dynamics discrepancy is the next-state gap |f_tar(s, a) - f_src(s, a)|
# hopper / halfcheetah / walker2d observe qpos[1:] + qvel (x position dropped), and the dynamics are x-invariant,
# so every probe state is recoverable from its observation with x = 0

GYM_IDS = dict(hopper = 'Hopper-v3', halfcheetah = 'HalfCheetah-v3', walker2d = 'Walker2d-v3')

def make_source_env(env_name):
    return gym.make(GYM_IDS[env_name.split('-')[0]])

def make_target_env(env_name, shift_level):
    return call_mujoco_env(dict(env_name = env_name, shift_level = shift_level))

def step_from(env, state, action):
    sim = env.unwrapped
    nq = sim.model.nq
    sim.set_state(np.concatenate(([0.], state[:nq - 1])), state[nq - 1:])
    obs, *_ = sim.step(action)
    return obs

def predict(env, states, actions):
    env.reset()
    return np.stack([step_from(env, s, a) for s, a in zip(states, actions)])

def dynamics_gap(src_env, tar_env, states, actions, *, scale):
    # scale: per-dimension std of next states so no dimension dominates the norm
    f_src, f_tar = predict(src_env, states, actions), predict(tar_env, states, actions)
    return np.linalg.norm((f_tar - f_src) / scale, axis = -1), f_src
