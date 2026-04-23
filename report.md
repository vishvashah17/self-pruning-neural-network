# Self-Pruning Neural Network Report

---

## 1. Why L1 Encourages Sparsity

The sparsity loss is defined as the mean of all gate values:

SparsityLoss = mean(sigmoid(gate_scores))

Since gate values are always positive (0 to 1), **L1 regularization** (sum or mean) pushes them toward zero with a constant gradient.

Unlike **L2 loss**, which weakens near zero, L1 continues applying pressure, causing many gates to become **exactly zero**. This effectively prunes the corresponding weights from the network.

---

## 2. Results

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
|:----------:|:-----------------:|:------------------:|
| 1.0        | 58.60             | 26.68              |
| 5.0        | 58.36             | 72.57              |
| 20.0       | 56.67             | 89.18              |

---

## 3. Analysis

As **λ increases**, sparsity increases significantly while accuracy decreases slightly:

- **λ = 1.0** — Low sparsity and highest accuracy.
- **λ = 5.0** — Best trade-off between sparsity and accuracy. 
- **λ = 20.0** — Very high sparsity with a slight drop in accuracy.

> This demonstrates that the network successfully learns to prune itself during training.

---

## 4. Gate Value Distribution

A successful pruning behavior is observed:

- **Spike near 0** — Represents pruned (inactive) weights.
- **Cluster away from 0** — Represents active weights.

📊 Reference: `gate_distribution.png`

---

## 5. Conclusion

The model effectively balances accuracy and sparsity, achieving up to **~89% sparsity** while maintaining competitive performance.

This confirms that **L1 regularization on gate parameters** enables dynamic self-pruning during training.
