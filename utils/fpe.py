"""Fixed-Parameter Expansion: expand-then-contract neuron splitting.

Each ``*ExpandThenContract`` class widens the hidden layer of a seed model by
``expansion_factor`` sub-neurons per original neuron, then rebalances sparsity
so the total nonzero parameter count matches the seed model's budget.

Two decisions parameterize the expansion:

* ``strategy`` (mask for the expanded fc1): ``"random"`` deals each original
  neuron's input features round-robin across its sub-neurons; ``"gram"`` first
  clusters input features by their fc1 weight-Gram-matrix similarity and then
  assigns whole clusters round-robin.
* Balancing (mask that prunes the excess weights created by expansion): fixed
  to the paper's ``"balanced"`` scheme -- drop the smallest-magnitude
  ``ceil(excess/2)`` entries in each of fc1 and (post-expansion) fc2.
"""

from __future__ import annotations

import copy
import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans


VALID_STRATEGIES = ("random", "gram")
_DEFAULT_GRAM_K = 32


# ---------------------------------------------------------------------------
# Mask construction helpers
# ---------------------------------------------------------------------------


def build_random_fc1_mask(
    hidden_size: int, in_features: int, expansion_factor: int
) -> torch.Tensor:
    """Block-diagonal fc1 mask: each of the ``hidden_size`` original neurons
    has its ``in_features`` inputs dealt round-robin (in a random order) across
    its ``expansion_factor`` sub-neurons.

    Returns a ``(expansion_factor * hidden_size, in_features)`` float mask with
    exactly ``hidden_size * in_features`` ones (same as the seed's fc1).
    """
    mask = torch.zeros(expansion_factor * hidden_size, in_features)
    for n in range(hidden_size):
        perm = np.random.permutation(in_features)
        for i, feat_idx in enumerate(perm):
            sub = i % expansion_factor
            mask[n * expansion_factor + sub, feat_idx] = 1.0
    return mask


def build_gram_fc1_mask(
    original_fc1_weight: torch.Tensor,
    expansion_factor: int,
    best_k: int = _DEFAULT_GRAM_K,
) -> torch.Tensor:
    """Block-diagonal fc1 mask that groups correlated input features together.

    Clusters the input features via KMeans on the weight-Gram matrix
    ``W.T @ W`` (reordered by average-linkage hierarchical clustering on
    ``1 - |G|``), then deals whole clusters round-robin across each original
    neuron's sub-neurons.
    """
    hidden_size, in_features = original_fc1_weight.shape
    W = original_fc1_weight.detach().cpu().numpy()
    G = W.T @ W
    D = 1.0 - np.abs(G)
    Z = linkage(squareform(D, checks=False), method="average")
    Z[Z < 0] = 0
    order = leaves_list(Z)
    G_reordered = G[np.ix_(order, order)]

    k = min(best_k, G_reordered.shape[0])
    labels = KMeans(n_clusters=k, n_init=100).fit(G_reordered).labels_
    order_2 = np.argsort(labels)
    boundary = np.where(np.diff(labels[order_2]) != 0)[0]

    clusters = []
    for c in range(k):
        start = 0 if c == 0 else boundary[c - 1] + 1
        end = G_reordered.shape[0] if c == k - 1 else boundary[c] + 1
        clusters.append(order[order_2][start:end])

    mask = torch.zeros(expansion_factor * hidden_size, in_features)
    for n in range(hidden_size):
        cluster_order = np.random.permutation(len(clusters))
        for i, cluster_idx in enumerate(cluster_order):
            sub = i % expansion_factor
            mask[n * expansion_factor + sub, clusters[cluster_idx]] = 1.0
    return mask


def build_balanced_masks(
    original_fc1_weight: torch.Tensor,
    original_fc2_weight: torch.Tensor,
    expansion_factor: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Magnitude-based pruning masks that remove the excess parameters
    introduced by expanding fc2 (whose input dim grows by ``expansion_factor``).

    Splits the excess evenly: drops the smallest ``ceil(excess/2)`` weights in
    the seed's fc1, and the smallest ``ceil(excess / (2 * expansion_factor))``
    weights in the seed's fc2 (each becomes ``expansion_factor`` dropped
    weights after ``repeat_interleave`` along dim=1).

    Returns ``(fc1_balance_mask, fc2_balance_mask)`` shaped to match the
    *expanded* fc1 and fc2 weights respectively.
    """
    hidden_size = original_fc1_weight.shape[0]
    num_classes = original_fc2_weight.shape[0]
    excess = (expansion_factor - 1) * hidden_size * num_classes

    fc1_mag = original_fc1_weight.abs().view(-1)
    fc2_mag = original_fc2_weight.abs().view(-1)
    threshold_fc1, _ = torch.kthvalue(fc1_mag, math.ceil(excess / 2))
    threshold_fc2, _ = torch.kthvalue(fc2_mag, math.ceil(excess / (2 * expansion_factor)))

    fc1_keep = (fc1_mag > threshold_fc1).view_as(original_fc1_weight)
    fc2_keep = (fc2_mag > threshold_fc2).view_as(original_fc2_weight)

    fc1_balance_mask = torch.repeat_interleave(fc1_keep, expansion_factor, dim=0).float()
    fc2_balance_mask = torch.repeat_interleave(fc2_keep, expansion_factor, dim=1).float()
    return fc1_balance_mask, fc2_balance_mask


def _build_fc1_disjoint_mask(
    original_fc1_weight: torch.Tensor,
    hidden_size: int,
    in_features: int,
    expansion_factor: int,
    strategy: str,
) -> torch.Tensor:
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"strategy must be one of {VALID_STRATEGIES}, got {strategy!r}")
    if strategy == "random":
        return build_random_fc1_mask(hidden_size, in_features, expansion_factor)
    return build_gram_fc1_mask(original_fc1_weight, expansion_factor)


# ---------------------------------------------------------------------------
# Expand-then-contract modules
# ---------------------------------------------------------------------------


class NeuralNetExpandThenContract(nn.Module):
    """Expand-then-contract MLP built from a :class:`NeuralNet` seed model.

    ``fc1`` is widened to ``expansion_factor * hidden_size`` outputs with a
    block-diagonal disjoint-input mask; ``fc2`` is widened to accept the wider
    hidden layer; both layers then have the ``build_balanced_masks`` pruning
    mask applied so total nonzero weights equal the seed's parameter budget.
    """

    def __init__(self, seed_model: nn.Module, expansion_factor: int, strategy: str = "random"):
        super().__init__()
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"strategy must be one of {VALID_STRATEGIES}, got {strategy!r}")

        self.expansion_factor = expansion_factor
        self.strategy = strategy
        self.input_size = seed_model.input_size
        self.hidden_size = seed_model.hidden_size
        self.num_classes = seed_model.num_classes

        seed_device = seed_model.fc1.weight.device
        orig_fc1 = seed_model.fc1.weight.data.clone()
        orig_fc2 = seed_model.fc2.weight.data.clone()

        self.fc1 = nn.Linear(self.input_size, expansion_factor * self.hidden_size, bias=False)
        self.fc1.weight.data = torch.repeat_interleave(orig_fc1, expansion_factor, dim=0)
        self.fc2 = nn.Linear(expansion_factor * self.hidden_size, self.num_classes, bias=False)
        self.fc2.weight.data = torch.repeat_interleave(orig_fc2, expansion_factor, dim=1)

        fc1_disjoint = _build_fc1_disjoint_mask(
            orig_fc1, self.hidden_size, self.input_size, expansion_factor, strategy
        )
        fc1_balance, fc2_balance = build_balanced_masks(orig_fc1, orig_fc2, expansion_factor)

        self.register_buffer("sparsity_mask_fc1", fc1_disjoint.to(seed_device))
        self.register_buffer("sparsity_mask_fc1_balance", fc1_balance.to(seed_device))
        self.register_buffer("sparsity_mask_fc2_balance", fc2_balance.to(seed_device))

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_size)
        w1 = self.fc1.weight * self.sparsity_mask_fc1 * self.sparsity_mask_fc1_balance
        w2 = self.fc2.weight * self.sparsity_mask_fc2_balance
        out = F.linear(x, w1)
        out = self.relu(out)
        out = F.linear(out, w2)
        return out


class NeuralNetEmbeddingExpandThenContract(nn.Module):
    """Same as :class:`NeuralNetExpandThenContract`, but preserves the seed
    model's ``embedding`` linear layer unchanged (only fc1/fc2 are expanded).
    """

    def __init__(self, seed_model: nn.Module, expansion_factor: int, strategy: str = "random"):
        super().__init__()
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"strategy must be one of {VALID_STRATEGIES}, got {strategy!r}")

        self.expansion_factor = expansion_factor
        self.strategy = strategy
        self.input_size = seed_model.input_size
        self.hidden_size = seed_model.hidden_size
        self.num_classes = seed_model.num_classes

        seed_device = seed_model.fc1.weight.device
        self.embedding = nn.Linear(self.input_size, self.input_size, bias=False)
        self.embedding.weight.data = seed_model.embedding.weight.data.clone()

        orig_fc1 = seed_model.fc1.weight.data.clone()
        orig_fc2 = seed_model.fc2.weight.data.clone()

        self.fc1 = nn.Linear(self.input_size, expansion_factor * self.hidden_size, bias=False)
        self.fc1.weight.data = torch.repeat_interleave(orig_fc1, expansion_factor, dim=0)
        self.fc2 = nn.Linear(expansion_factor * self.hidden_size, self.num_classes, bias=False)
        self.fc2.weight.data = torch.repeat_interleave(orig_fc2, expansion_factor, dim=1)

        fc1_disjoint = _build_fc1_disjoint_mask(
            orig_fc1, self.hidden_size, self.input_size, expansion_factor, strategy
        )
        fc1_balance, fc2_balance = build_balanced_masks(orig_fc1, orig_fc2, expansion_factor)

        self.register_buffer("sparsity_mask_fc1", fc1_disjoint.to(seed_device))
        self.register_buffer("sparsity_mask_fc1_balance", fc1_balance.to(seed_device))
        self.register_buffer("sparsity_mask_fc2_balance", fc2_balance.to(seed_device))

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_size)
        x = self.embedding(x)
        w1 = self.fc1.weight * self.sparsity_mask_fc1 * self.sparsity_mask_fc1_balance
        w2 = self.fc2.weight * self.sparsity_mask_fc2_balance
        out = F.linear(x, w1)
        out = self.relu(out)
        out = F.linear(out, w2)
        return out


class CNNMLPExpandThenContract(nn.Module):
    """Same as :class:`NeuralNetExpandThenContract`, but built from a
    :class:`CNNMLP` seed model: the convolutional backbone is deep-copied
    unchanged and only the fc1/fc2 classifier head is expanded.
    """

    def __init__(self, seed_model: nn.Module, expansion_factor: int, strategy: str = "random"):
        super().__init__()
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"strategy must be one of {VALID_STRATEGIES}, got {strategy!r}")

        self.expansion_factor = expansion_factor
        self.strategy = strategy
        self.flatten_size = seed_model.flatten_size
        self.hidden_size = seed_model.hidden_size
        self.num_classes = seed_model.num_classes

        seed_device = seed_model.fc1.weight.device
        self.conv_layers = copy.deepcopy(seed_model.conv_layers)

        orig_fc1 = seed_model.fc1.weight.data.clone()
        orig_fc2 = seed_model.fc2.weight.data.clone()

        self.fc1 = nn.Linear(self.flatten_size, expansion_factor * self.hidden_size, bias=False)
        self.fc1.weight.data = torch.repeat_interleave(orig_fc1, expansion_factor, dim=0)
        self.fc2 = nn.Linear(expansion_factor * self.hidden_size, self.num_classes, bias=False)
        self.fc2.weight.data = torch.repeat_interleave(orig_fc2, expansion_factor, dim=1)

        fc1_disjoint = _build_fc1_disjoint_mask(
            orig_fc1, self.hidden_size, self.flatten_size, expansion_factor, strategy
        )
        fc1_balance, fc2_balance = build_balanced_masks(orig_fc1, orig_fc2, expansion_factor)

        self.register_buffer("sparsity_mask_fc1", fc1_disjoint.to(seed_device))
        self.register_buffer("sparsity_mask_fc1_balance", fc1_balance.to(seed_device))
        self.register_buffer("sparsity_mask_fc2_balance", fc2_balance.to(seed_device))

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        w1 = self.fc1.weight * self.sparsity_mask_fc1 * self.sparsity_mask_fc1_balance
        w2 = self.fc2.weight * self.sparsity_mask_fc2_balance
        out = F.linear(x, w1)
        out = self.relu(out)
        out = F.linear(out, w2)
        return out
