from __future__ import annotations

from torch import cat
from torch.nn import Module

from adaptation.networks import MLP

# D_SA(s, a): logit ≈ log p_t(s, a) / p_s(s, a)

class OccupancyClassifier(Module):
    def __init__(self, dim_state, dim_action, *, dim_hidden = 256):
        super().__init__()
        self.net = MLP(dim_state + dim_action, dim_hidden = dim_hidden)

    def forward(self, state, action):
        return self.net(cat((state, action), dim = -1))
