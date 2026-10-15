from __future__ import annotations

from collections import namedtuple

from torch import nn, cat
from torch.nn import Module

from adaptation.networks import MLP

# f_joint(s, a, C) = f_occ(s, a) + g(s, a, embed(C))
# f_occ trained on (s, a) labels, g is then the local dynamics residual ≈ log p_t(C | s, a) / p_s(C | s, a)
# one shared classifier across clusters instead of K separate ones, so N_t / K is not the bottleneck

JointLogits = namedtuple('JointLogits', ['joint', 'occ'])

class JointClusterClassifier(Module):
    def __init__(self, dim_state, dim_action, num_clusters, *, dim_cluster = 16, dim_hidden = 256):
        super().__init__()
        self.cluster_embed = nn.Embedding(num_clusters, dim_cluster)
        self.occ = MLP(dim_state + dim_action, dim_hidden = dim_hidden)
        self.residual = MLP(dim_state + dim_action + dim_cluster, dim_hidden = dim_hidden)

    def forward(self, state, action, cluster):
        sa = cat((state, action), dim = -1)
        occ = self.occ(sa)
        joint = occ.detach() + self.residual(cat((sa, self.cluster_embed(cluster)), dim = -1))
        return JointLogits(joint, occ)

    def dynamics_log_ratio(self, state, action, cluster):
        joint, occ = self.forward(state, action, cluster)
        return joint - occ
