from __future__ import annotations

import torch
from torch import cat
from torch.nn import Module
import torch.nn.functional as F

from adaptation.networks import MLP, sample_batch
from adaptation.joint_cluster_classifier import JointLogits

# normalized conditional ratio
# p_t(k | z) = p_s(k | z) exp(g(z, k)) / sum_j p_s(j | z) exp(g(z, j))
# so l(z, k) = g(z, k) - log sum_j p_s(j | z) exp(g(z, j)) satisfies sum_k p_s(k | z) exp(l(z, k)) = 1 by construction:
# any component of g that depends on z alone cancels in the normalizer, so occupancy the occ branch did not fit cannot leak into l
# p_s(k | z) is a frozen k-way classifier fit on source transitions only

class ClusterPredictor(Module):
    def __init__(self, dim_state, dim_action, num_clusters, *, dim_hidden = 256):
        super().__init__()
        self.net = MLP(dim_state + dim_action, num_clusters, dim_hidden = dim_hidden)

    def forward(self, state, action):
        return self.net(cat((state, action), dim = -1)).log_softmax(-1)

def train_cluster_predictor(pred, src, *, steps = 3000, batch_size = 256, lr = 3e-4):
    device = next(pred.parameters()).device
    opt = torch.optim.Adam(pred.parameters(), lr = lr)
    for _ in range(steps):
        s, a, c = (x.to(device) for x in sample_batch(src, batch_size))
        loss = F.nll_loss(pred(s, a), c)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return pred

class NormalizedClusterClassifier(Module):
    def __init__(self, dim_state, dim_action, num_clusters, predictor, *, dim_hidden = 256):
        super().__init__()
        self.occ = MLP(dim_state + dim_action, dim_hidden = dim_hidden)
        self.tilt = MLP(dim_state + dim_action, num_clusters, dim_hidden = dim_hidden)
        self.predictor = predictor.requires_grad_(False)

    def dynamics_log_ratio(self, state, action, cluster):
        g = self.tilt(cat((state, action), dim = -1))                      # (n, k)
        log_norm = torch.logsumexp(self.predictor(state, action) + g, dim = -1, keepdim = True)
        return g.gather(-1, cluster[:, None]) - log_norm                     # (n, 1)

    def forward(self, state, action, cluster):
        occ = self.occ(cat((state, action), dim = -1))
        return JointLogits(occ.detach() + self.dynamics_log_ratio(state, action, cluster), occ)

# the plain two-classifier difference: logit D_ZK(z, k) - logit D_Z(z), each trained on its own, no shared structure

class ClusterConditionedClassifier(Module):
    def __init__(self, dim_state, dim_action, num_clusters, *, dim_cluster = 16, dim_hidden = 256):
        super().__init__()
        self.cluster_embed = torch.nn.Embedding(num_clusters, dim_cluster)
        self.net = MLP(dim_state + dim_action + dim_cluster, dim_hidden = dim_hidden)

    def forward(self, state, action, cluster):
        return self.net(cat((state, action, self.cluster_embed(cluster)), dim = -1))
