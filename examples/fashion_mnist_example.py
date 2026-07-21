"""FashionMNIST comparison: NeuralNet baseline vs NeuralNetExpandThenContract.

Trains a plain 2-layer MLP for 50 epochs total. For the expanded variants, the
seed model is trained for 25 epochs, then split into an `expansion_factor`-wider
version (parameter budget preserved), then trained for the remaining 25 epochs.
Reports best test accuracy for the baseline plus both fc1 sparsity strategies.

FashionMNIST is downloaded on first run to ./data/ (~30MB).
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

from utils.fpe import NeuralNetExpandThenContract
from utils.models import NeuralNet
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

INPUT_SIZE = 28 * 28
NUM_CLASSES = 10
DATA_ROOT = "./data"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading FashionMNIST (downloading to {DATA_ROOT} if not present)...")
    transform = transforms.ToTensor()
    train_dataset = torchvision.datasets.FashionMNIST(
        root=DATA_ROOT, train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.FashionMNIST(
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
    print(f"\n[base] NeuralNet, {INITIAL_EPOCHS} seed epochs")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    base = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
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
    print(f"FashionMNIST | hidden={HIDDEN_SIZE}, ef={EXPANSION_FACTOR}, {TOTAL_EPOCHS} total epochs")
    print("=" * 68)
    print(f"  Baseline NeuralNet                             : {max(baseline_test_acc):6.2f}%")
    print(f"  Expand-then-Contract (random) split @ ep {INITIAL_EPOCHS}    : {max(random_test_acc):6.2f}%")
    print(f"  Expand-then-Contract (gram)   split @ ep {INITIAL_EPOCHS}    : {max(gram_test_acc):6.2f}%")


if __name__ == "__main__":
    main()
