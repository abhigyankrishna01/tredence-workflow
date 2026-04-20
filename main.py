import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), -2.0))

        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in = in_features
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)


class PrunableNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(32 * 32 * 3, 256)
        self.relu = nn.ReLU()
        self.fc2 = PrunableLinear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_sparsity_loss(model: nn.Module) -> torch.Tensor:
    loss = 0.0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            loss = loss + torch.sigmoid(module.gate_scores).sum()
    return loss


def get_all_gates(model: nn.Module) -> torch.Tensor:
    gates = []
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates.append(torch.sigmoid(module.gate_scores).detach().view(-1))
    return torch.cat(gates)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = 100.0 * correct / total
    gates = get_all_gates(model)
    total_weights = gates.numel()
    zero_weights = (gates < 1e-2).sum().item()
    print(f"Total Weights: {total_weights}, Pruned: {zero_weights}")
    sparsity = 100.0 * (gates < 1e-2).float().mean().item()
    return acc, sparsity, gates.cpu().numpy()


def train_one_experiment(
    lambda_sparse: float,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
):
    model = PrunableNet().to(device)
    gate_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "gate_scores" in name:
            gate_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.Adam(
        [
            {"params": other_params, "lr": 1e-3},
            {"params": gate_params, "lr": 0.0035},
        ]
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            ce_loss = criterion(logits, y)
            sparse_loss = get_sparsity_loss(model)
            loss = ce_loss + lambda_sparse * sparse_loss
            loss.backward()
            optimizer.step()

    return evaluate(model, test_loader, device)


def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path("data")
    batch_size = 64
    epochs = 2
    lambdas = [0.0001, 0.001, 0.01]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)

    results = []
    all_gate_values = []

    for lam in lambdas:
        acc, sparsity, gates = train_one_experiment(lam, train_loader, test_loader, device, epochs)
        results.append((lam, acc, sparsity))
        all_gate_values.append(gates)
        print(f"Lambda: {lam} -> Accuracy: {acc:.2f}%, Sparsity: {sparsity:.2f}%")

    with open("results.txt", "w", encoding="utf-8") as f:
        f.write("Lambda,Accuracy,Sparsity\n")
        for lam, acc, sparsity in results:
            f.write(f"{lam},{acc:.4f},{sparsity:.4f}\n")

    gate_values = np.concatenate(all_gate_values)
    plt.figure(figsize=(8, 5))
    plt.hist(gate_values, bins=50)
    plt.xlabel("Gate value")
    plt.ylabel("Count")
    plt.title("Distribution of gate values")
    plt.tight_layout()
    plt.savefig("gate_distribution.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
