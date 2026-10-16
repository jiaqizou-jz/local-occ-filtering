import sys, os, numpy as np, torch, gym, yaml
sys.path.insert(0, '/home/jz520/local-occ-filtering'); sys.path.insert(0, '/home/jz520/local-occ-filtering/algo'); os.chdir('/home/jz520/local-occ-filtering')
from offline_offline.iql import Policy
from envs.mujoco.call_mujoco_env import call_mujoco_env

hid = yaml.safe_load(open('config/mujoco/score_iql/hopper.yaml'))['hidden_sizes']
run_dir = 'logs/gate_o/keep0.5/{arm}/score_iql/hopper-friction-srcdatatype-medium-replay-tardatatype-medium-2.0/r{seed}/models/model_actor'

def rollout(actor, env, n = 5, seed = 0):
    env.seed(seed); out = []
    for ep in range(n):
        s, done, ret, t, vx, zmin = env.reset(), False, 0., 0, [], 9.
        while not done:
            with torch.no_grad():
                _, _, mean = actor(torch.tensor(s, dtype = torch.float32).view(1, -1))
            s, rew, done, _ = env.step(mean.squeeze(0).numpy()); ret += rew; t += 1; vx.append(s[5]); zmin = min(zmin, s[0])
        out.append((ret, t, np.mean(vx), zmin, s[1]))
    return out

src_env = gym.make('Hopper-v3'); tar_env = call_mujoco_env({'env_name': 'hopper-friction', 'shift_level': 2.0})
print('arm/seed            env   ret per episode                              len per episode              mean vx   min z  final angle')
for arm, seed in (('concat', 0), ('concat', 1), ('random', 0), ('oracle_bellman', 1), ('oracle_bellman', 0), ('oracle_state', 1)):
    actor = Policy(11, 3, 1., hidden_size = hid); actor.load_state_dict(torch.load(run_dir.format(arm = arm, seed = seed), map_location = 'cpu')); actor.eval()
    nan = any(torch.isnan(p).any() for p in actor.parameters())
    for name, env in (('src', src_env), ('tar', tar_env)):
        o = rollout(actor, env)
        print(f'{arm + "/" + str(seed):<18}  {name}   {" ".join(f"{r:7.1f}" for r, *_ in o)}   {" ".join(f"{t:5d}" for _, t, *_ in o)}   {np.mean([v for _, _, v, *_ in o]):6.2f}   {np.mean([z for *_, z, _ in o]):5.2f}  {np.mean([ang for *_, ang in o]):6.2f}' + ('  NAN PARAMS' if nan else ''))
