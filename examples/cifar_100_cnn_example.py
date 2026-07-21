"""CIFAR-100 (raw pixels) comparison: CNNMLP baseline vs
CNNMLPExpandThenContract.

Trains a small CNN backbone with a two-layer MLP head end-to-end on raw
CIFAR-100 pixels. Follows the same train-then-split-then-continue protocol
as the other examples: 25 seed epochs, then split into an expanded variant
(only the fc1/fc2 head is expanded; the conv backbone is deep-copied
unchanged) and train the remaining 25 epochs. Parameter budget of the head
is preserved.

CIFAR-100 downloads to ``./data/`` on first run (~170MB). Training is
compute-heavy; expect ~10-30 minutes on GPU depending on hardware.
"""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

from utils.fpe import CNNMLPExpandThenContract
from utils.models import CNNMLP
from utils.training import train_model


# --- Config ---------------------------------------------------------------
SEED = 0
FLATTEN_SIZE = 256
HIDDEN_SIZE = 16
EXPANSION_FACTOR = 4
TOTAL_EPOCHS = 50
INITIAL_EPOCHS = 25
POST_SPLIT_EPOCHS = TOTAL_EPOCHS - INITIAL_EPOCHS
LEARNING_RATE = 1e-3
BATCH_SIZE = 64

NUM_CLASSES = 100
DATA_ROOT = "./data"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading CIFAR-100 (downloading to {DATA_ROOT} if not present)...")
    transform = transforms.ToTensor()
    train_dataset = torchvision.datasets.CIFAR100(
        root=DATA_ROOT, train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=DATA_ROOT, train=False, transform=transform, download=True
    )

    def to_device_dataset(dataset: torch.utils.data.Dataset) -> TensorDataset:
        loader = DataLoader(dataset, batch_size=len(dataset))
        images, labels = next(iter(loader))
        return TensorDataset(images.to(device), labels.to(device))

    train_loader = DataLoader(
        to_device_dataset(train_dataset), batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        to_device_dataset(test_dataset), batch_size=BATCH_SIZE, shuffle=False
    )

    # 1. Base: train a single model for INITIAL_EPOCHS. This shared prefix is
    #    then reused three ways so all runs are identical for the first
    #    INITIAL_EPOCHS.
    print(f"\n[base] CNNMLP, {INITIAL_EPOCHS} seed epochs")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    base = CNNMLP(FLATTEN_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
    _, _, _, base_test_acc = train_model(
        base, LEARNING_RATE, INITIAL_EPOCHS, train_loader, test_loader
    )

    # 2. Snapshot the base before it diverges. The duplicate is split; the
    #    original is trained straight through to the full budget.
    shared_seed = copy.deepcopy(base)

    # 3. Baseline: continue the original for the remaining epochs (no split).
    print(f"\n[baseline] CNNMLP, {POST_SPLIT_EPOCHS} more epochs ({TOTAL_EPOCHS} total)")
    _, _, _, baseline_cont_test_acc = train_model(
        base, LEARNING_RATE, POST_SPLIT_EPOCHS, train_loader, test_loader
    )
    baseline_test_acc = base_test_acc + baseline_cont_test_acc

    # 4. Expand-then-contract with random fc1 mask (from the duplicate).
    print(
        f"\n[random] CNNMLPExpandThenContract"
        f"(expansion_factor={EXPANSION_FACTOR}, strategy='random'), "
        f"{POST_SPLIT_EPOCHS} more epochs"
    )
    random_expanded = CNNMLPExpandThenContract(
        shared_seed, EXPANSION_FACTOR, strategy="random"
    ).to(device)
    _, _, _, random_test_acc = train_model(
        random_expanded, LEARNING_RATE, POST_SPLIT_EPOCHS, train_loader, test_loader
    )

    # 5. Expand-then-contract with gram-clustered fc1 mask (from the duplicate).
    print(
        f"\n[gram] CNNMLPExpandThenContract"
        f"(expansion_factor={EXPANSION_FACTOR}, strategy='gram'), "
        f"{POST_SPLIT_EPOCHS} more epochs"
    )
    gram_expanded = CNNMLPExpandThenContract(
        shared_seed, EXPANSION_FACTOR, strategy="gram"
    ).to(device)
    _, _, _, gram_test_acc = train_model(
        gram_expanded, LEARNING_RATE, POST_SPLIT_EPOCHS, train_loader, test_loader
    )

    print()
    print("=" * 68)
    print(
        f"CIFAR-100 (raw pixels) | flatten={FLATTEN_SIZE}, hidden={HIDDEN_SIZE}, "
        f"ef={EXPANSION_FACTOR}, {TOTAL_EPOCHS} total epochs"
    )
    print("=" * 68)
    print(f"  Baseline CNNMLP                                : {max(baseline_test_acc):6.2f}%")
    print(f"  Expand-then-Contract (random) split @ ep {INITIAL_EPOCHS}    : {max(random_test_acc):6.2f}%")
    print(f"  Expand-then-Contract (gram)   split @ ep {INITIAL_EPOCHS}    : {max(gram_test_acc):6.2f}%")


if __name__ == "__main__":
    main()
