"""Boolean k-DNF comparison: NeuralNetEmbedding baseline vs
NeuralNetEmbeddingExpandThenContract.

Trains a small MLP-with-embedding for 50 epochs total. For the expanded
variants, the seed model is trained for 25 epochs, then split into an
`expansion_factor`-wider version (with the parameter budget preserved), then
trained for the remaining 25 epochs. Reports best test accuracy for the
baseline plus both fc1 sparsity strategies.

Hidden size is deliberately smaller than the number of clauses (8 hidden vs
16 clauses), so the seed model is capacity-limited and expansion should help.
"""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.boolean_data import generate_data
from utils.fpe import NeuralNetEmbeddingExpandThenContract
from utils.models import NeuralNetEmbedding
from utils.training import train_model


# --- Config ---------------------------------------------------------------
SEED = 0
HIDDEN_SIZE = 8
EXPANSION_FACTOR = 4
TOTAL_EPOCHS = 50
INITIAL_EPOCHS = 25
POST_SPLIT_EPOCHS = TOTAL_EPOCHS - INITIAL_EPOCHS
LEARNING_RATE = 1e-3
BATCH_SIZE = 64

# Boolean task: 16 disjoint clauses of size 4 => 64 literals; a sample is
# positive iff at least one clause is fully satisfied.
M_LITERALS = 64
K_LITERALS_PER_CLAUSE = 4
MIN_ONES, MAX_ONES = 4, 8
N_TRAIN, N_TEST = 4000, 1000
NUM_CLASSES = 2


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(
        f"Generating boolean data: {N_TRAIN} train / {N_TEST} test, "
        f"{M_LITERALS} literals, {M_LITERALS // K_LITERALS_PER_CLAUSE} clauses of size {K_LITERALS_PER_CLAUSE}"
    )
    X_train, y_train = generate_data(
        N_TRAIN, M_LITERALS, K_LITERALS_PER_CLAUSE, MIN_ONES, MAX_ONES, seed=SEED
    )
    X_test, y_test = generate_data(
        N_TEST, M_LITERALS, K_LITERALS_PER_CLAUSE, MIN_ONES, MAX_ONES, seed=SEED + 1
    )
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # 1. Base: train a single model for INITIAL_EPOCHS. This shared prefix is
    #    then reused three ways so all runs are identical for the first
    #    INITIAL_EPOCHS.
    print(f"\n[base] NeuralNetEmbedding, {INITIAL_EPOCHS} seed epochs")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    base = NeuralNetEmbedding(M_LITERALS, HIDDEN_SIZE, NUM_CLASSES).to(device)
    _, _, _, base_test_acc = train_model(
        base, LEARNING_RATE, INITIAL_EPOCHS, train_loader, test_loader
    )

    # 2. Snapshot the base before it diverges. The duplicate is split; the
    #    original is trained straight through to the full budget.
    shared_seed = copy.deepcopy(base)

    # 3. Baseline: continue the original for the remaining epochs (no split).
    print(f"\n[baseline] NeuralNetEmbedding, {POST_SPLIT_EPOCHS} more epochs ({TOTAL_EPOCHS} total)")
    _, _, _, baseline_cont_test_acc = train_model(
        base, LEARNING_RATE, POST_SPLIT_EPOCHS, train_loader, test_loader
    )
    baseline_test_acc = base_test_acc + baseline_cont_test_acc

    # 4. Expand-then-contract with random fc1 mask (from the duplicate).
    print(
        f"\n[random] NeuralNetEmbeddingExpandThenContract"
        f"(expansion_factor={EXPANSION_FACTOR}, strategy='random'), "
        f"{POST_SPLIT_EPOCHS} more epochs"
    )
    random_expanded = NeuralNetEmbeddingExpandThenContract(
        shared_seed, EXPANSION_FACTOR, strategy="random"
    ).to(device)
    _, _, _, random_test_acc = train_model(
        random_expanded, LEARNING_RATE, POST_SPLIT_EPOCHS, train_loader, test_loader
    )

    # 5. Expand-then-contract with gram-clustered fc1 mask (from the duplicate).
    print(
        f"\n[gram] NeuralNetEmbeddingExpandThenContract"
        f"(expansion_factor={EXPANSION_FACTOR}, strategy='gram'), "
        f"{POST_SPLIT_EPOCHS} more epochs"
    )
    gram_expanded = NeuralNetEmbeddingExpandThenContract(
        shared_seed, EXPANSION_FACTOR, strategy="gram"
    ).to(device)
    _, _, _, gram_test_acc = train_model(
        gram_expanded, LEARNING_RATE, POST_SPLIT_EPOCHS, train_loader, test_loader
    )

    print()
    print("=" * 68)
    print(
        f"Boolean k-DNF | m={M_LITERALS} literals, k={K_LITERALS_PER_CLAUSE}, "
        f"hidden={HIDDEN_SIZE}, ef={EXPANSION_FACTOR}, {TOTAL_EPOCHS} total epochs"
    )
    print("=" * 68)
    print(f"  Baseline NeuralNetEmbedding                    : {max(baseline_test_acc):6.2f}%")
    print(f"  Expand-then-Contract (random) split @ ep {INITIAL_EPOCHS}    : {max(random_test_acc):6.2f}%")
    print(f"  Expand-then-Contract (gram)   split @ ep {INITIAL_EPOCHS}    : {max(gram_test_acc):6.2f}%")


if __name__ == "__main__":
    main()
