import numpy as np
import torch

from offline_offline.iql import IQL
from adaptation.clustering import kmeans, assign
from adaptation.networks import train_domain_classifier
from adaptation.joint_cluster_classifier import JointClusterClassifier
from adaptation.score import local_log_ratio
from adaptation.filtering import quantile_mask

# iql backbone + localized occupancy-corrected source filtering
# source buffer is scored once with log p_t(C | s, a) / p_s(C | s, a), C = kmeans(s') fit on target, then hard-filtered in place
# discrepancy estimation lives in adaptation/, this file only wires it to the rl loop

def buffer_arrays(buffer):
    return tuple(getattr(buffer, k)[:buffer.size] for k in ('state', 'action', 'next_state', 'reward', 'not_done'))

class LocalOccIQL(IQL):

    def __init__(self, config, device):
        super().__init__(config, device)
        self.num_clusters = config.get('num_clusters', 10)
        self.classifier_steps = config.get('classifier_steps', 20000)
        self.keep_fraction = config.get('keep_fraction', 0.5)
        self.filtered = False

    def filter_source(self, src_replay_buffer, tar_replay_buffer, writer = None):
        src = buffer_arrays(src_replay_buffer)
        tar = buffer_arrays(tar_replay_buffer)

        centroids = kmeans(tar[2], self.num_clusters)
        src_c, tar_c = assign(src[2], centroids), assign(tar[2], centroids)

        f32 = lambda x: torch.tensor(x, dtype = torch.float32)
        src_t = (f32(src[0]), f32(src[1]), torch.tensor(src_c))
        tar_t = (f32(tar[0]), f32(tar[1]), torch.tensor(tar_c))

        joint = JointClusterClassifier(self.config['state_dim'], self.config['action_dim'], self.num_clusters, dim_hidden = self.config['hidden_sizes']).to(self.device)
        train_domain_classifier(joint, src_t, tar_t, steps = self.classifier_steps, batch_size = 2 * self.config['batch_size'], lr = self.config['actor_lr'])

        log_ratio = torch.cat([local_log_ratio(joint, *(x[i:i + 4096].to(self.device) for x in src_t)).cpu() for i in range(0, len(src_c), 4096)])
        mask = quantile_mask(log_ratio, keep = self.keep_fraction).numpy()

        for k, v in zip(('state', 'action', 'next_state', 'reward', 'not_done'), src):
            setattr(src_replay_buffer, k, v[mask])
        src_replay_buffer.size, src_replay_buffer.ptr = int(mask.sum()), 0
        self.filtered = True

        if writer is not None:
            writer.add_scalar('filter/source kept', src_replay_buffer.size, 0)
            writer.add_scalar('filter/log ratio mean', log_ratio.mean(), 0)

    def train(self, src_replay_buffer, tar_replay_buffer, batch_size = 128, writer = None):
        if not self.filtered:
            self.filter_source(src_replay_buffer, tar_replay_buffer, writer)
        super().train(src_replay_buffer, tar_replay_buffer, batch_size, writer)
