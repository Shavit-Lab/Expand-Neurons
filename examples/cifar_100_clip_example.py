"""CIFAR-100 (via CLIP embeddings) comparison: NeuralNet baseline vs
NeuralNetExpandThenContract.

Loads (or precomputes and caches) CLIP ViT-B/32 image embeddings for
CIFAR-100 via ``utils.process_clip_embeddings``, then trains a small MLP
head on top. Follows the same train-then-split-then-continue protocol as
the other examples: 25 seed epochs, then split into an expanded variant and
train the remaining 25 epochs; parameter budget preserved.

First run downloads CLIP (~600MB) and CIFAR-100 (~170MB) into ``./data/``
and caches the resulting embeddings; subsequent runs skip both downloads.
"""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.fpe import NeuralNetExpandThenContract
from utils.models import NeuralNet
from utils.process_clip_embeddings import get_cifar100_clip_embeddings
from utils.training import train_model


# --- Config ---------------------------------------------------------------
SEED = 0
HIDDEN_SIZE = 16
EXPANSION_FACTOR = 4
TOTAL_EPOCHS = 50
INITIAL_EPOCHS = 25
POST_SPLIT_EPOCHS = TOTAL_EPOCHS - INITIAL_EPOCHS
LEARNING_RATE = 1e-3
BATCH_SIZE = 64

NUM_CLASSES = 100


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = get_cifar100_clip_embeddings(device=device)
    train_emb = data["train_embeddings"].to(device)
    train_labels = data["train_labels"].to(device)
    test_emb = data["test_embeddings"].to(device)
    test_labels = data["test_labels"].to(device)

    input_size = train_emb.size(1)
    print(f"CLIP embedding dim: {input_size}")

    train_loader = DataLoader(
        TensorDataset(train_emb, train_labels), batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(test_emb, test_labels), batch_size=BATCH_SIZE, shuffle=False
    )

    # 1. Base: train a single model for INITIAL_EPOCHS. This shared prefix is
    #    then reused three ways so all runs are identical for the first
    #    INITIAL_EPOCHS.
    print(f"\n[base] NeuralNet, {INITIAL_EPOCHS} seed epochs")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    base = NeuralNet(input_size, HIDDEN_SIZE, NUM_CLASSES).to(device)
    _, _, _, base_test_acc = train_model(
        base, LEARNING_RATE, INITIAL_EPOCHS, train_loader, test_loader
    )

    # 2. Snapshot the base before it diverges. The duplicate is split; the
    #    original is trained straight through to the full budget.
    shared_seed = copy.deepcopy(base)

    # 3. Baseline: continue the original for the remaining epochs (no split).
    print(f"\n[baseline] NeuralNet, {POST_SPLIT_EPOCHS} more epochs ({TOTAL_EPOCHS} total)")
    _, _, _, baseline_cont_test_acc = train_model(
        base, LEARNING_RATE, POST_SPLIT_EPOCHS, train_loader, test_loader
    )
    baseline_test_acc = base_test_acc + baseline_cont_test_acc

    # 4. Expand-then-contract with random fc1 mask (from the duplicate).
    print(
        f"\n[random] NeuralNetExpandThenContract"
        f"(expansion_factor={EXPANSION_FACTOR}, strategy='random'), "
        f"{POST_SPLIT_EPOCHS} more epochs"
    )
    random_expanded = NeuralNetExpandThenContract(
        shared_seed, EXPANSION_FACTOR, strategy="random"
    ).to(device)
    _, _, _, random_test_acc = train_model(
        random_expanded, LEARNING_RATE, POST_SPLIT_EPOCHS, train_loader, test_loader
    )

    # 5. Expand-then-contract with gram-clustered fc1 mask (from the duplicate).
    print(
        f"\n[gram] NeuralNetExpandThenContract"
        f"(expansion_factor={EXPANSION_FACTOR}, strategy='gram'), "
        f"{POST_SPLIT_EPOCHS} more epochs"
    )
    gram_expanded = NeuralNetExpandThenContract(
        shared_seed, EXPANSION_FACTOR, strategy="gram"
    ).to(device)
    _, _, _, gram_test_acc = train_model(
        gram_expanded, LEARNING_RATE, POST_SPLIT_EPOCHS, train_loader, test_loader
    )

    print()
    print("=" * 68)
    print(
        f"CIFAR-100 CLIP embeddings | hidden={HIDDEN_SIZE}, ef={EXPANSION_FACTOR}, "
        f"{TOTAL_EPOCHS} total epochs"
    )
    print("=" * 68)
    print(f"  Baseline NeuralNet                             : {max(baseline_test_acc):6.2f}%")
    print(f"  Expand-then-Contract (random) split @ ep {INITIAL_EPOCHS}    : {max(random_test_acc):6.2f}%")
    print(f"  Expand-then-Contract (gram)   split @ ep {INITIAL_EPOCHS}    : {max(gram_test_acc):6.2f}%")


if __name__ == "__main__":
    main()
