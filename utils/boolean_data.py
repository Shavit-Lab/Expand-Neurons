"""Synthetic k-DNF boolean-clause data used by the paper's toy embedding
experiments.

The formula partitions ``m_literals`` inputs into
``m_literals // k_literals_per_clause`` fixed disjoint clauses of size
``k_literals_per_clause``; a sample is positive iff at least one clause has all
of its literals set to 1.
"""

from typing import Optional, Tuple
import random

import numpy as np
import torch


def generate_data(
    n_samples: int,
    m_literals: int,
    k_literals_per_clause: int,
    minimum_ones: int,
    maximum_ones: int,
    pos_proportion: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a labelled k-DNF dataset.

    Positive samples satisfy at least one of the fixed k-clauses (literals
    ``i*k .. i*k+k-1`` all true for some ``i``); negative samples satisfy none.
    Each sample has between ``minimum_ones`` and ``maximum_ones`` active
    literals.

    Returns:
        X: ``(n_samples, m_literals)`` float32 tensor of 0/1 features.
        y: ``(n_samples,)`` int64 tensor of 0/1 labels.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    num_pos = int(n_samples * pos_proportion)
    num_neg = n_samples - num_pos
    pos_data = np.zeros((num_pos, m_literals))
    neg_data = np.zeros((num_neg, m_literals))
    pos_labels = np.ones(num_pos)
    neg_labels = np.zeros(num_neg)
    num_clauses = m_literals // k_literals_per_clause

    for i in range(num_pos):
        clause_index = np.random.randint(0, num_clauses)
        num_ones = np.random.randint(minimum_ones, maximum_ones + 1)
        literal_indices = [clause_index * k_literals_per_clause + j
                           for j in range(k_literals_per_clause)]
        while len(literal_indices) < num_ones:
            literal_index = np.random.randint(0, m_literals)
            if literal_index not in literal_indices:
                literal_indices.append(literal_index)
        for literal_index in literal_indices:
            pos_data[i, literal_index] = 1
        assert evaluate_sample(pos_data[i], m_literals, k_literals_per_clause) == 1

    num_negative_samples = 0
    unsatisfied_num_ones = -1
    while num_negative_samples < num_neg:
        if np.random.rand() < 0.5:
            clause_index = np.random.randint(0, num_clauses)
            if unsatisfied_num_ones == -1:
                num_ones = np.random.randint(minimum_ones, maximum_ones + 1)
            else:
                num_ones = unsatisfied_num_ones
            literal_indices = [clause_index * k_literals_per_clause + j
                               for j in range(k_literals_per_clause)]
            while len(literal_indices) < num_ones + 1:
                literal_index = np.random.randint(0, m_literals)
                if literal_index not in literal_indices:
                    literal_indices.append(literal_index)
            for literal_index in literal_indices:
                neg_data[num_negative_samples, literal_index] = 1
            incorrect_index = np.random.randint(0, k_literals_per_clause)
            neg_data[num_negative_samples, literal_indices[incorrect_index]] = 0
        else:
            if unsatisfied_num_ones == -1:
                num_ones = np.random.randint(minimum_ones, maximum_ones + 1)
            else:
                num_ones = unsatisfied_num_ones
            literal_indices = np.random.choice(range(m_literals), num_ones, replace=False)
            for literal_index in literal_indices:
                neg_data[num_negative_samples, literal_index] = 1

        if evaluate_sample(neg_data[num_negative_samples], m_literals, k_literals_per_clause) == 0:
            num_negative_samples += 1
            unsatisfied_num_ones = -1
        else:
            neg_data[num_negative_samples, :] = 0
            unsatisfied_num_ones = num_ones

    X = torch.tensor(np.concatenate((pos_data, neg_data), axis=0), dtype=torch.float32)
    y = torch.tensor(np.concatenate((pos_labels, neg_labels), axis=0), dtype=torch.long)
    return X, y


def evaluate_sample(sample: np.ndarray, m_literals: int, k_literals_per_clause: int) -> int:
    """Return 1 if any of the fixed k-clauses is fully satisfied by ``sample``, else 0."""
    num_clauses = m_literals // k_literals_per_clause
    for i in range(num_clauses):
        clause = sample[i * k_literals_per_clause:(i + 1) * k_literals_per_clause]
        if np.sum(clause) == k_literals_per_clause:
            return 1
    return 0
