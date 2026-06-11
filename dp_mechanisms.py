"""Pure differential privacy mechanisms."""

import numpy as np


def exponential_mechanism(scores, epsilon, sensitivity=1.0, rng=None):
    """Select index proportional to exp(eps * score / (2*sens)). eps-DP."""
    rng = rng or np.random.default_rng()
    scores = np.asarray(scores, dtype=np.float64)
    log_w = (epsilon * scores) / (2.0 * sensitivity)
    log_w -= log_w.max()
    w = np.exp(log_w)
    return int(rng.choice(len(scores), p=w / w.sum()))


def joint_exponential_mechanism(scores, k, epsilon, sensitivity=1.0, rng=None):
    """Select top-k via iterated EM. Each pick uses eps/k; total = eps."""
    rng = rng or np.random.default_rng()
    scores = np.asarray(scores, dtype=np.float64)
    per_eps = epsilon / k
    selected, mask = [], np.ones(len(scores), dtype=bool)
    for _ in range(min(k, len(scores))):
        masked = np.where(mask, scores, -np.inf)
        idx = exponential_mechanism(masked, per_eps, sensitivity, rng)
        selected.append(idx)
        mask[idx] = False
    return selected


def laplace_mechanism(value, epsilon, sensitivity=1.0, rng=None):
    """Add Laplace noise. eps-DP."""
    rng = rng or np.random.default_rng()
    return float(value + rng.laplace(0, sensitivity / epsilon))


def randomized_response(true_label, num_classes, epsilon, rng=None):
    """Randomized response for categorical data. eps-LDP."""
    rng = rng or np.random.default_rng()
    e_eps = np.exp(epsilon)
    if rng.random() < e_eps / (e_eps + num_classes - 1):
        return true_label
    others = [i for i in range(num_classes) if i != true_label]
    return int(rng.choice(others))
