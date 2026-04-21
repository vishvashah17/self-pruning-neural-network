"""
Self-Pruning Neural Network on CIFAR-10
========================================
Tredence AI Engineering Intern - Case Study

This script implements a feed-forward neural network for CIFAR-10 image
classification that learns to prune itself during training via learnable
gate parameters and L1 sparsity regularization.

Architecture:
    - Custom PrunableLinear layers with sigmoid-gated weights
    - Sparsity regularization: Total Loss = CE + lambda * mean(all gates)
    - Three lambda values compared to show sparsity-accuracy trade-off

Key design decisions:
    - gate_scores initialized to +5.0 so sigmoid(5) ~ 0.993 (gates start fully OPEN)
    - SparsityLoss = MEAN of all gate values  (normalized, scale-independent)
    - Gates use a higher learning rate than weights so they respond to sparsity pressure
    - Lambda values [1.0, 5.0, 20.0] are meaningful because loss is in (0,1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  
import os
import json
from typing import List, Tuple, Dict


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: PrunableLinear Layer
# ─────────────────────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that learns to prune its own weights.

    Each weight w_ij has a paired scalar gate g_ij in (0, 1).
    The effective weight used in computation is:  w_ij * g_ij

    When g_ij → 0 the weight is effectively removed (pruned).
    When g_ij → 1 the weight is fully active.

    The gate is produced by:  g_ij = sigmoid(score_ij)
    where score_ij (gate_scores) is a learned parameter updated by the optimizer.

    Initialization:
        gate_scores = +5.0  →  sigmoid(5) ≈ 0.993
        All gates start near 1 (fully open). The L1 sparsity penalty then
        selectively drives unimportant gates toward 0 during training.

    Gradient flow:
        Both self.weight and self.gate_scores receive gradients via autograd
        because all operations (sigmoid, multiply, F.linear) are differentiable.

    Args:
        in_features  (int): number of input features
        out_features (int): number of output features
    """

    def __init__(self, in_features: int, out_features: int):
        super(PrunableLinear, self).__init__()

        self.in_features  = in_features
        self.out_features = out_features

        # ── Standard parameters ──────────────────────────────────────────────
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # ── Gate scores ──────────────────────────────────────────────────────
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), fill_value=5.0)
        )

        # Kaiming uniform init for weights (good default for ReLU networks)
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
            Step 1 — gates        = sigmoid(gate_scores)     in (0, 1)
            Step 2 — pruned_w     = weight * gates           element-wise
            Step 3 — output       = F.linear(x, pruned_w, bias)

        Gradients flow correctly through both weight and gate_scores.
        """
        gates          = torch.sigmoid(self.gate_scores)    
        pruned_weights = self.weight * gates                 
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return gate values detached from the computation graph (for metrics)."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity_level(self, threshold: float = 1e-2) -> float:
        """Fraction of this layer's weights whose gate is below threshold."""
        gates = self.get_gates()
        return (gates < threshold).float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Network Definition
# ─────────────────────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 built entirely from PrunableLinear layers.

    CIFAR-10 input: 32 x 32 x 3 = 3072 features, 10 output classes.

    Architecture:
        Flatten → PrunableLinear(3072→1024) → BN → ReLU → Dropout(0.3)
               → PrunableLinear(1024→512)  → BN → ReLU → Dropout(0.3)
               → PrunableLinear(512→256)   → BN → ReLU → Dropout(0.3)
               → PrunableLinear(256→10)    → logits

    BatchNorm is included after each prunable layer to stabilize training when
    many weights are zeroed out by pruning.
    """

    def __init__(self, dropout_rate: float = 0.3):
        super(SelfPruningNet, self).__init__()

        self.flatten = nn.Flatten()

        # All linear layers are prunable
        self.fc1 = PrunableLinear(3072, 1024)
        self.fc2 = PrunableLinear(1024,  512)
        self.fc3 = PrunableLinear(512,   256)
        self.fc4 = PrunableLinear(256,    10)

        # Batch normalization after each hidden layer
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)                                      
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))      
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))         
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))       
        x = self.fc4(x)                                       
        return x

    def prunable_layers(self) -> List[PrunableLinear]:
        """Return all PrunableLinear layers in the model."""
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def compute_sparsity_loss(self) -> torch.Tensor:
        """
        PART 2: Sparsity Regularization Loss

        SparsityLoss = mean( sigmoid(gate_scores) )  over ALL gates in ALL layers

        Why MEAN instead of SUM:
            The raw sum of ~4 million gates would be ~2,000,000, completely
            overwhelming the CE loss (~2.3). Using the MEAN keeps SparsityLoss
            in (0, 1), making it comparable in scale to CE and making lambda
            values intuitive and tunable.

        Why L1 (mean of gate values) encourages sparsity:
            gates = sigmoid(scores) are always positive, so |gate| = gate.
            The gradient of mean(gates) w.r.t. each gate is a constant 1/N.
            This constant gradient continues pushing gates toward 0 even
            when they are already very small — unlike L2 whose gradient
            vanishes near zero and can never force values to exactly 0.
        """
        all_gates = []
        for layer in self.prunable_layers():
            all_gates.append(torch.sigmoid(layer.gate_scores).view(-1))
        return torch.cat(all_gates).mean()

    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        """
        Overall fraction of weights effectively pruned (gate < threshold).
        A gate below 1e-2 contributes less than 1 percent of its weight.
        """
        total, pruned = 0, 0
        for layer in self.prunable_layers():
            gates   = layer.get_gates()
            pruned += (gates < threshold).sum().item()
            total  += gates.numel()
        return pruned / total if total > 0 else 0.0

    def all_gate_values(self) -> np.ndarray:
        """Flat numpy array of all gate values — used for plotting."""
        values = []
        for layer in self.prunable_layers():
            values.append(layer.get_gates().cpu().numpy().flatten())
        return np.concatenate(values)


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-10 train and test sets.

    Train: RandomCrop + RandomHorizontalFlip + Normalize  (standard augmentation)
    Test : Normalize only
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.CIFAR10(
        root='./data', train=True,  download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=False
    )
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:         SelfPruningNet,
    loader:        DataLoader,
    optimizer:     optim.Optimizer,
    device:        torch.device,
    lambda_sparse: float,
) -> Tuple[float, float, float]:
    """
    Train for one epoch using:
        Total Loss = CrossEntropyLoss + lambda * SparsityLoss

    The optimizer updates ALL parameters including gate_scores via one
    backward pass. Gradients flow through F.linear and the element-wise
    multiply back to both weight and gate_scores tensors.

    Returns:
        avg_total_loss, avg_ce_loss, avg_sparsity_loss
    """
    model.train()
    total_sum, ce_sum, sp_sum = 0.0, 0.0, 0.0
    n_batches = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logits  = model(images)
        ce_loss = F.cross_entropy(logits, labels)
        sp_loss = model.compute_sparsity_loss()         
        loss    = ce_loss + lambda_sparse * sp_loss     

        loss.backward()      
        optimizer.step()

        total_sum += loss.item()
        ce_sum    += ce_loss.item()
        sp_sum    += sp_loss.item()
        n_batches += 1

    return total_sum / n_batches, ce_sum / n_batches, sp_sum / n_batches


def evaluate(
    model:  SelfPruningNet,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute classification accuracy on a DataLoader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds    = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Full Experiment for One Lambda Value
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    lambda_sparse: float,
    train_loader:  DataLoader,
    test_loader:   DataLoader,
    device:        torch.device,
    epochs:        int   = 30,
    lr_weights:    float = 1e-3,
    lr_gates:      float = 1e-2,
    weight_decay:  float = 1e-4,
) -> Dict:
    """
    Train SelfPruningNet with the given lambda and return results dict.

    Two separate parameter groups:
        weights / biases / BN : lr = lr_weights (1e-3)
        gate_scores            : lr = lr_gates   (1e-2, 10x higher)

    Gates need higher LR to overcome their strong positive initialization (+5.0)
    and be pushed below the prune threshold in a reasonable number of epochs.

    Returns dict: lambda, test_accuracy, sparsity_level, gate_values, history
    """
    print(f"\n{'='*60}")
    print(f"  Training with lambda = {lambda_sparse}")
    print(f"{'='*60}")

    model = SelfPruningNet(dropout_rate=0.3).to(device)

    gate_params   = [p for n, p in model.named_parameters() if 'gate_scores' in n]
    weight_params = [p for n, p in model.named_parameters() if 'gate_scores' not in n]

    optimizer = optim.Adam([
        {'params': weight_params, 'lr': lr_weights, 'weight_decay': weight_decay},
        {'params': gate_params,   'lr': lr_gates,   'weight_decay': 0.0},
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "total_loss": [], "cls_loss": [], "spr_loss": [],
        "test_acc":   [], "sparsity":  []
    }

    for epoch in range(1, epochs + 1):
        total_l, cls_l, spr_l = train_one_epoch(
            model, train_loader, optimizer, device, lambda_sparse
        )
        test_acc = evaluate(model, test_loader, device)
        sparsity = model.overall_sparsity(threshold=1e-2)
        scheduler.step()

        history["total_loss"].append(total_l)
        history["cls_loss"].append(cls_l)
        history["spr_loss"].append(spr_l)
        history["test_acc"].append(test_acc)
        history["sparsity"].append(sparsity)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"CE: {cls_l:.4f} | "
                f"Sparsity Loss: {spr_l:.4f} | "
                f"Test Acc: {test_acc*100:.2f}% | "
                f"Sparsity: {sparsity*100:.1f}%"
            )

    final_acc      = evaluate(model, test_loader, device)
    final_sparsity = model.overall_sparsity(threshold=1e-2)
    gate_values    = model.all_gate_values()

    print(f"\n  Final Test Accuracy : {final_acc*100:.2f}%")
    print(f"  Final Sparsity Level: {final_sparsity*100:.2f}%")

    return {
        "lambda":         lambda_sparse,
        "test_accuracy":  final_acc,
        "sparsity_level": final_sparsity,
        "gate_values":    gate_values,
        "history":        history,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_gate_distribution(results: List[Dict], output_dir: str = "."):
    """
    Gate value histogram for each lambda.
    Successful result: large spike near 0 AND a separate cluster near 0.5-1.0
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    for ax, res, color in zip(axes, results, colors):
        gates = res["gate_values"]
        ax.hist(gates, bins=100, color=color, alpha=0.85, edgecolor='white', linewidth=0.3)
        ax.set_title(
            f"lambda = {res['lambda']}\n"
            f"Acc: {res['test_accuracy']*100:.1f}%  |  "
            f"Sparsity: {res['sparsity_level']*100:.1f}%",
            fontsize=11, fontweight='bold'
        )
        ax.set_xlabel("Gate Value", fontsize=10)
        ax.set_ylabel("Count",      fontsize=10)
        ax.axvline(
            x=0.01, color='black', linestyle='--',
            linewidth=1.2, label='Prune threshold (0.01)'
        )
        ax.legend(fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        "Gate Value Distributions - Self-Pruning Neural Network\n"
        "Spike near 0 = pruned weights | Cluster away from 0 = active weights",
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()

    path = os.path.join(output_dir, "gate_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Gate distribution plot saved -> {path}")


def plot_training_curves(results: List[Dict], output_dir: str = "."):
    """Plot test accuracy and sparsity over training epochs for each lambda."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    for res, color in zip(results, colors):
        label   = f"lambda={res['lambda']}"
        history = res["history"]
        epochs  = range(1, len(history["test_acc"]) + 1)
        ax1.plot(epochs, [a * 100 for a in history["test_acc"]], color=color, label=label, linewidth=2)
        ax2.plot(epochs, [s * 100 for s in history["sparsity"]],  color=color, label=label, linewidth=2)

    ax1.set_title("Test Accuracy vs Epochs",  fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Test Accuracy (%)")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.set_title("Sparsity Level vs Epochs", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Sparsity (%)")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training curves saved      -> {path}")


def print_results_table(results: List[Dict]):
    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    print(f"  {'Lambda':<12} {'Test Accuracy':>16} {'Sparsity Level':>16}")
    print("-"*60)
    for res in results:
        print(
            f"  {res['lambda']:<12} "
            f"{res['test_accuracy']*100:>14.2f}%  "
            f"{res['sparsity_level']*100:>14.2f}%"
        )
    print("="*60)


def save_results_json(results: List[Dict], output_dir: str = "."):
    """Save final metrics to results.json."""
    data = [
        {
            "lambda":         r["lambda"],
            "test_accuracy":  round(r["test_accuracy"]  * 100, 2),
            "sparsity_level": round(r["sparsity_level"] * 100, 2),
        }
        for r in results
    ]
    path = os.path.join(output_dir, "results.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results JSON saved         -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Hyperparameters ────────────────────────────────────────────────────
    EPOCHS       = 30
    BATCH_SIZE   = 128
    LR_WEIGHTS   = 1e-3       
    LR_GATES     = 1e-2       
    WEIGHT_DECAY = 1e-4
    OUTPUT_DIR   = "."

    # Lambda values for sparsity-accuracy trade-off comparison.
    LAMBDAS = [1.0, 5.0, 20.0]

    # ── Device ─────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"\n  Using device: {device}")

    # ── Data ───────────────────────────────────────────────────────────────
    print("\n  Loading CIFAR-10 ...")
    train_loader, test_loader = get_cifar10_loaders(BATCH_SIZE)

    # ── Run all three experiments ───────────────────────────────────────────
    all_results = []
    for lam in LAMBDAS:
        result = run_experiment(
            lambda_sparse = lam,
            train_loader  = train_loader,
            test_loader   = test_loader,
            device        = device,
            epochs        = EPOCHS,
            lr_weights    = LR_WEIGHTS,
            lr_gates      = LR_GATES,
            weight_decay  = WEIGHT_DECAY,
        )
        all_results.append(result)

    # ── Output ─────────────────────────────────────────────────────────────
    print_results_table(all_results)
    save_results_json(all_results, OUTPUT_DIR)
    plot_gate_distribution(all_results, OUTPUT_DIR)
    plot_training_curves(all_results, OUTPUT_DIR)

    print("\n  All done! Files saved: gate_distribution.png, training_curves.png, results.json\n")


if __name__ == "__main__":
    main()