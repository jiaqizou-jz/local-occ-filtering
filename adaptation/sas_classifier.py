from __future__ import annotations

from torch import cat
from torch.nn import Module

from adaptation.networks import MLP

# D_SAS(s, a, s'): logit ≈ log p_t(s, a, s') / p_s(s, a, s')
# darc / dara global dynamics ratio is logit D_SAS - logit D_SA

class DynamicsClassifier(Module):
    def __init__(self, dim_state, dim_action, *, dim_hidden = 256):
        super().__init__()
        self.net = MLP(2 * dim_state + dim_action, dim_hidden = dim_hidden)

    def forward(self, state, action, next_state):
        return self.net(cat((state, action, next_state), dim = -1))
