from __future__ import annotations

from toy.linear_gaussian import Domain, rollout

# four-cell experiment matrix
# closed loop gain A + B K must stay inside the unit circle for a stationary occupancy

HORIZON = 50

SOURCE = Domain(A = 0.9, B = 0.5, K = -0.8, sigma = 0.1, sigma_a = 0.3)

CASES = dict(
    A = SOURCE,                                   # same dynamics, same behavior
    B = SOURCE._replace(K = 0.),                  # same dynamics, different behavior   <- lodada false positive
    C = SOURCE._replace(A = 0.6, B = 0.8),        # different dynamics, same behavior
    D = SOURCE._replace(A = 0.6, B = 0.8, K = 0.) # different dynamics, different behavior
)

def make_case(name, *, num_source = 50_000, num_target = 5_000, horizon = HORIZON, seed = 0):
    src, tar = SOURCE, CASES[name]
    src_data = rollout(src, num_traj = num_source // horizon, horizon = horizon, seed = seed)
    tar_data = rollout(tar, num_traj = num_target // horizon, horizon = horizon, seed = seed + 1)
    return src_data, tar_data, src, tar
