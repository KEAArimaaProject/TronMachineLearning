"""Helpers to convert between Stable-Baselines3 / PyTorch policy state_dicts
and the GA genome vector used by trainer.MLPPolicy.

This module provides:
- state_dict_to_genome(state_dict, obs_dim, hidden, action_dim=3)
- genome_to_state_dict(genome, state_dict_template, obs_dim, hidden, action_dim=3)
- load_genome_from_ppo(checkpoint_path, obs_dim, hidden, action_dim=3)
- create_population_from_base(base_genome, pop_size, rng, noise_std)

The conversion is robust: it matches linear layers by shape and handles
PyTorch weight layout (out_features, in_features) vs the GA layout
(w1 stored as (in, out)).
"""
from collections import OrderedDict
from pathlib import Path
from typing import OrderedDict as ODType

import numpy as np

try:
    import torch
    from stable_baselines3 import PPO
except Exception:
    # Defer import errors until functions that need them are called
    torch = None  # type: ignore
    PPO = None  # type: ignore


def state_dict_to_genome(state_dict: ODType, obs_dim: int, hidden: int, action_dim: int = 3) -> np.ndarray:
    """Convert a PyTorch state_dict (policy) to a flattened genome vector.

    Looks for two linear layers matching shapes for the feature transform and
    action head and orders parameters as MLPPolicy expects: w1 (obs_dim*hidden),
    b1 (hidden), w2 (hidden*action_dim), b2 (action_dim).
    """
    # Collect weight/bias arrays
    weights = {}
    biases = {}
    for k, v in state_dict.items():
        if k.endswith(".weight"):
            weights[k] = v.detach().cpu().numpy()
        elif k.endswith(".bias"):
            biases[k] = v.detach().cpu().numpy()

    w1 = None; b1 = None; w2 = None; b2 = None

    # Find feature layer (matching obs_dim and hidden)
    for k_w, arr in weights.items():
        # arr shape may be (hidden, obs_dim) or (obs_dim, hidden)
        if arr.shape == (hidden, obs_dim):
            w1 = arr.T.copy()  # to (obs_dim, hidden)
            prefix = k_w[:-len('.weight')]
            b1 = biases.get(prefix + '.bias', np.zeros(hidden, dtype=np.float32)).astype(np.float32)
            break
        if arr.shape == (obs_dim, hidden):
            w1 = arr.copy()
            prefix = k_w[:-len('.weight')]
            b1 = biases.get(prefix + '.bias', np.zeros(hidden, dtype=np.float32)).astype(np.float32)
            break

    # Find action layer
    for k_w, arr in weights.items():
        if arr.shape == (action_dim, hidden):
            w2 = arr.T.copy()  # to (hidden, action_dim)
            prefix = k_w[:-len('.weight')]
            b2 = biases.get(prefix + '.bias', np.zeros(action_dim, dtype=np.float32)).astype(np.float32)
            break
        if arr.shape == (hidden, action_dim):
            w2 = arr.copy()
            prefix = k_w[:-len('.weight')]
            b2 = biases.get(prefix + '.bias', np.zeros(action_dim, dtype=np.float32)).astype(np.float32)
            break

    if w1 is None or w2 is None:
        raise RuntimeError("Could not locate compatible linear layers in provided state_dict.\n"
                           "Inspect state_dict keys/shapes to ensure the policy uses a single hidden layer.")

    genome = np.concatenate([w1.ravel(), b1.ravel(), w2.ravel(), b2.ravel()]).astype(np.float32)
    return genome


def genome_to_state_dict(genome: np.ndarray, state_dict_template: ODType, obs_dim: int, hidden: int, action_dim: int = 3) -> ODType:
    """Return a new state_dict (OrderedDict) based on template with actor layers replaced
    by parameters unpacked from genome. Non-matching tensors are left unchanged.
    """
    import torch as _torch

    i = 0
    n1 = obs_dim * hidden
    w1 = genome[i:i+n1].reshape(obs_dim, hidden).astype(np.float32); i += n1
    b1 = genome[i:i+hidden].astype(np.float32); i += hidden
    n2 = hidden * action_dim
    w2 = genome[i:i+n2].reshape(hidden, action_dim).astype(np.float32); i += n2
    b2 = genome[i:i+action_dim].astype(np.float32)

    new_state = OrderedDict()
    for k, v in state_dict_template.items():
        if k.endswith('.weight'):
            arr = v.cpu().numpy()
            if arr.shape == (hidden, obs_dim):
                new_state[k] = _torch.from_numpy(w1.T)
            elif arr.shape == (obs_dim, hidden):
                new_state[k] = _torch.from_numpy(w1)
            elif arr.shape == (action_dim, hidden):
                new_state[k] = _torch.from_numpy(w2.T)
            elif arr.shape == (hidden, action_dim):
                new_state[k] = _torch.from_numpy(w2)
            else:
                new_state[k] = v
        elif k.endswith('.bias'):
            arr = v.cpu().numpy()
            if arr.shape == (hidden,):
                new_state[k] = _torch.from_numpy(b1)
            elif arr.shape == (action_dim,):
                new_state[k] = _torch.from_numpy(b2)
            else:
                new_state[k] = v
        else:
            new_state[k] = v

    return new_state


def load_genome_from_ppo(checkpoint_path: str, obs_dim: int, hidden: int, action_dim: int = 3) -> np.ndarray:
    """Load a PPO model from `checkpoint_path` and return a genome vector matching
    MLPPolicy ordering.
    """
    if PPO is None or torch is None:
        raise RuntimeError("stable_baselines3 and torch are required to load PPO checkpoints.")

    path = str(checkpoint_path)
    model = PPO.load(path)
    sd = model.policy.state_dict()
    return state_dict_to_genome(sd, obs_dim=obs_dim, hidden=hidden, action_dim=action_dim)


def create_population_from_base(base_genome: np.ndarray, pop_size: int, rng: np.random.Generator, noise_std: float = 0.02) -> np.ndarray:
    """Create a population by duplicating base_genome and adding gaussian noise.
    Returns array of shape (pop_size, len(base_genome)).
    """
    base = np.asarray(base_genome, dtype=np.float32)
    shape = (pop_size, base.shape[0])
    noise = rng.normal(0, noise_std, size=shape).astype(np.float32)
    return np.tile(base, (pop_size, 1)).astype(np.float32) + noise

