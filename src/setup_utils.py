import json

import os
import robustbench as rob
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


def move_model_extras_to_device(model, dev):
    for attr in ("mean", "std"):
        if hasattr(model, attr):
            t = getattr(model, attr)
            if torch.is_tensor(t):
                setattr(model, attr, t.to(dev))
    return model


def device_for_model(name: str) -> str:
    if hasattr(torch.backends, "cuda") and torch.cuda.is_available():
        print(f"CUDA available for {name}")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"MPS is available for {name}")
        return "mps"
    else:
        print(f"no MPS or CUDA for {name}. Fallback to CPU")
        return "cpu"


def compute_elapsed_time(start, end) -> list:
    elapsed = end - start
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    return [int(hours), int(minutes), seconds]


def load_data(dataset_samples: int, seed: int, batch_size: int):
    print("\n---- LOADING DATA ----")
    transform = transforms.ToTensor()
    print("Loading CIFAR10 dataset...")
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(testset), dataset_samples, replace=False)

    print(f"Generating Subset with:\n"
          f" - samples: {dataset_samples}\n"
          f" - seed: {seed}\n"
          f" - batch_size: {batch_size}")
    subset = Subset(testset, idx)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)

    x_list, y_list = [], []
    for x, y in loader:
        x_list.append(x)
        y_list.append(y)

    x = torch.cat(x_list, dim=0)  # stay on CPU
    y = torch.cat(y_list, dim=0)  # stay on CPU
    print("Dataset loaded on CPU!")
    return x, y


def load_models(model_names: list) -> dict:
    print("\n---- LOADING MODELS ----")
    models = {}
    for name in model_names:
        print(f"Loading {name}...")
        models[name] = rob.load_model(model_name=name, dataset="cifar10", threat_model="Linf")
        models[name].eval()
    print("Models loaded!")
    return models
