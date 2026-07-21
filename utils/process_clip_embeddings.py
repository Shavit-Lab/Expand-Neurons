"""Precompute CLIP ViT-B/32 image embeddings for a torchvision image dataset
so downstream training can operate on cached vectors instead of running the
vision transformer on every step.

Provides a low-level ``compute_clip_embeddings`` helper and a convenience
``get_cifar100_clip_embeddings`` that loads from ``./data/`` if cached and
computes-and-caches otherwise. Also runnable as a CLI:

    python -m utils.process_clip_embeddings

which populates ``./data/CIFAR100_clip_embeddings.pkl``.
"""

from typing import Optional, Tuple
import os
import pickle

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_DATA_ROOT = "./data"
DEFAULT_CACHE_PATH = "./data/CIFAR100_clip_embeddings.pkl"


def compute_clip_embeddings(
    dataset: Dataset,
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(embeddings, labels)`` for every item in ``dataset``.

    Embeddings are L2-normalized 512-dim vectors from CLIP-ViT-B/32; labels
    are whatever the dataset's ``__getitem__`` returns as the second element.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_embeddings, all_labels = [], []
    clip_model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="CLIP embed"):
            inputs = processor(images=images, return_tensors="pt", do_rescale=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            feats = clip_model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeddings.append(feats.cpu())
            all_labels.append(labels)
    return torch.cat(all_embeddings), torch.cat(all_labels)


def get_cifar100_clip_embeddings(
    data_root: str = DEFAULT_DATA_ROOT,
    cache_path: str = DEFAULT_CACHE_PATH,
    clip_model_name: str = DEFAULT_CLIP_MODEL,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> dict:
    """Load CIFAR-100 CLIP embeddings from disk, computing-and-caching if
    the cache file is missing.

    Returns a dict with keys ``train_embeddings``, ``train_labels``,
    ``test_embeddings``, ``test_labels`` -- all CPU tensors.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached CLIP embeddings from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Cache miss: computing CLIP embeddings (device={device})")
    print(f"Loading CLIP model: {clip_model_name}")
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)

    print(f"Loading CIFAR-100 to {data_root}")
    train_ds = torchvision.datasets.CIFAR100(
        root=data_root, train=True, transform=transforms.ToTensor(), download=True
    )
    test_ds = torchvision.datasets.CIFAR100(
        root=data_root, train=False, transform=transforms.ToTensor(), download=True
    )

    print("Embedding train set...")
    train_emb, train_labels = compute_clip_embeddings(
        train_ds, clip_model, processor, device, batch_size
    )
    print("Embedding test set...")
    test_emb, test_labels = compute_clip_embeddings(
        test_ds, clip_model, processor, device, batch_size
    )

    result = {
        "train_embeddings": train_emb,
        "train_labels": train_labels,
        "test_embeddings": test_emb,
        "test_labels": test_labels,
    }

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved embeddings to {cache_path}")
    print(f"  train: {tuple(train_emb.shape)}, test: {tuple(test_emb.shape)}")

    return result


if __name__ == "__main__":
    get_cifar100_clip_embeddings()
