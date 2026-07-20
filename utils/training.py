from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    learning_rate: float,
    num_epochs: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    l1_lambda: float = 0.0,
    l2_lambda: float = 0.0,
    device: Optional[torch.device] = None,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Train ``model`` for ``num_epochs`` with Adam and cross-entropy loss.

    If ``model`` exposes ``sparsity_mask_fc1`` / ``sparsity_mask_fc1_balance``
    (i.e. is one of the ``*ExpandThenContract`` variants), the L1 penalty is
    computed against the masked-effective fc1 weights.

    Returns four per-epoch lists: train losses, train accuracies (%), test
    losses, test accuracies (%).
    """
    if device is None:
        device = next(model.parameters()).device

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    is_expanded = hasattr(model, "sparsity_mask_fc1")

    train_losses: List[float] = []
    train_accuracies: List[float] = []
    test_losses: List[float] = []
    test_accuracies: List[float] = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if l2_lambda > 0:
                loss = loss + l2_lambda * sum(p.pow(2).sum() for p in model.parameters())
            if l1_lambda > 0:
                if is_expanded:
                    fc1_effective = (
                        model.fc1.weight
                        * model.sparsity_mask_fc1
                        * model.sparsity_mask_fc1_balance
                    )
                else:
                    fc1_effective = model.fc1.weight
                loss = loss + l1_lambda * fc1_effective.abs().sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss, train_acc = evaluate_model(model, train_loader, device=device)
        test_loss, test_acc = evaluate_model(model, test_loader, device=device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"train_loss={running_loss / len(train_loader):.4f} "
            f"train_acc={train_acc:.2f}% test_acc={test_acc:.2f}%"
        )

    return train_losses, train_accuracies, test_losses, test_accuracies


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """Return ``(mean_loss, accuracy_percent)`` for ``model`` on ``data_loader``."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return total_loss / len(data_loader), 100.0 * correct / total
