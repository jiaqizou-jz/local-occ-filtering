import numpy as np
import h5py
import torch

from offline_offline.iql import IQL

# iql backbone + source selection from precomputed per-transition scores (gate o)
# source: concat (all source data) | target_only (both halves of every batch from the target) | a key of score_file, keep the top keep_fraction
# target_file: transition-level target hdf5 that replaces the loaded target buffer (call_tar_dataset rebuilds transitions from
#   consecutive observations, which is wrong for paired transition sets)
# source_weight: relative loss weight lambda, L = (1 - lambda) mean_target + lambda mean_source for v / q / actor alike;
#   1/2 is the plain concat mean over the 128 + 128 batch, 0 keeps only the 128 target samples of each batch

KEYS = ('state', 'action', 'next_state', 'reward', 'not_done')

def load_target(path):
    with h5py.File(path, 'r') as h:
        s, a, ns, r, d = (h[k][:].astype(np.float32) for k in ('observations', 'actions', 'next_observations', 'rewards', 'terminals'))
    return s, a, ns, r.reshape(-1, 1), 1. - d.reshape(-1, 1)

def set_buffer(buffer, arrays):
    for k, v in zip(KEYS, arrays):
        setattr(buffer, k, v)
    buffer.size, buffer.ptr = len(arrays[0]), 0

def arrays_of(buffer):
    return tuple(getattr(buffer, k)[:buffer.size] for k in KEYS)

class ScoreIQL(IQL):

    def __init__(self, config, device):
        super().__init__(config, device)
        self.source = config.get('source', 'concat')
        self.keep_fraction = config.get('keep_fraction', 0.5)
        self.score_file = config.get('score_file')
        self.target_file = config.get('target_file')
        self.target_subset = config.get('target_subset')     # n_t sweep: keep this many target transitions
        self.subset_seed = config.get('subset_seed', 0)      # one fixed permutation per seed -> nested subsets d_500 c d_1000 c d_2000 c d_5000
        self.bc = config.get('bc', False)                    # actor = plain action mse (advantage weight 1); q / v still trained but unused by the actor
        self.source_weight = config.get('source_weight')    # lambda of the relative loss weight; None = plain concat path of the parent
        self.prepared = False

    def prepare(self, src_replay_buffer, tar_replay_buffer, writer = None):
        if self.target_file:
            set_buffer(tar_replay_buffer, load_target(self.target_file))

        if self.target_subset:
            # index complete transition tuples of the loaded buffer: s' / not_done keep odrl's semantics, nothing is rebuilt from neighbours
            idx = np.random.default_rng(self.subset_seed).permutation(tar_replay_buffer.size)[:self.target_subset]
            set_buffer(tar_replay_buffer, tuple(v[idx] for v in arrays_of(tar_replay_buffer)))

        if self.source == 'target_only':
            set_buffer(src_replay_buffer, arrays_of(tar_replay_buffer))
        elif self.source != 'concat':
            score = np.load(self.score_file)[self.source]
            assert len(score) == src_replay_buffer.size, (len(score), src_replay_buffer.size)
            keep = score >= np.quantile(score, 1. - self.keep_fraction)
            set_buffer(src_replay_buffer, tuple(v[keep] for v in arrays_of(src_replay_buffer)))

        print(f'source {self.source}: {src_replay_buffer.size} transitions, target {tar_replay_buffer.size}' + (f' (subset seed {self.subset_seed})' if self.target_subset else '') + (', actor bc (advantage weight 1)' if self.bc else '') + (f', source weight {self.source_weight}' if self.source_weight is not None else ''), flush = True)
        if writer is not None:
            writer.add_scalar('filter/source kept', src_replay_buffer.size, 0)
        self.prepared = True

    def update_policy(self, advantage_batch, state_batch, action_batch):
        if not self.bc:
            return super().update_policy(advantage_batch, state_batch, action_batch)
        return self.policy.bc_loss(state_batch, action_batch).mean()

    def train(self, src_replay_buffer, tar_replay_buffer, batch_size = 128, writer = None):
        if not self.prepared:
            self.prepare(src_replay_buffer, tar_replay_buffer, writer)
        if self.source_weight is None:
            return super().train(src_replay_buffer, tar_replay_buffer, batch_size, writer)
        self.train_weighted(src_replay_buffer, tar_replay_buffer, batch_size, writer)

    def train_weighted(self, src_replay_buffer, tar_replay_buffer, batch_size, writer):
        # same three updates as iql.train, each loss replaced by (1 - lambda) mean over the target half + lambda mean over the source half
        self.total_it += 1
        lam = self.source_weight
        src = src_replay_buffer.sample(batch_size)
        tar = tar_replay_buffer.sample(batch_size)
        state, action, next_state, reward, not_done = (torch.cat([s, t], 0) for s, t in zip(src, tar))
        w = torch.cat([torch.full((batch_size, 1), lam / batch_size), torch.full((batch_size, 1), (1. - lam) / batch_size)], 0).to(state.device)

        with torch.no_grad():
            q_t = torch.min(*self.target_q_funcs(state, action))
        v = self.v_func(state)
        adv = q_t - v
        v_loss = (w * torch.abs(self.lam - (adv < 0).float()) * adv ** 2).sum()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        with torch.no_grad():
            value_target = reward + not_done * self.discount * self.v_func(next_state)
        q_1, q_2 = self.q_funcs(state, action)
        q_loss = (w * ((q_1 - value_target) ** 2 + (q_2 - value_target) ** 2)).sum()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        self.update_target()

        for p in self.q_funcs.parameters():
            p.requires_grad = False
        exp_adv = torch.exp(self.temp * adv.detach()).clamp(max = 100.) if not self.bc else 1.
        pi_loss = (w * exp_adv * self.policy.bc_loss(state, action).mean(-1, keepdim = True)).sum()
        self.policy_optimizer.zero_grad()
        pi_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()
        for p in self.q_funcs.parameters():
            p.requires_grad = True

        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/adv', adv.mean(), self.total_it)
            writer.add_scalar('train/value', v.mean(), self.total_it)
            writer.add_scalar('train/q1', q_1.mean(), self.total_it)
