# Self-Pruning Neural Network

> A minimal, efficient implementation of a self-pruning neural network using learnable gating and L1 sparsity regularization.

## What it does

- Uses a custom `PrunableLinear` layer with learnable `gate_scores`
- Computes effective weights as `weight * sigmoid(gate_scores)`
- Trains on CIFAR-10 with combined objective:
  - Classification: CrossEntropyLoss
  - Sparsity: L1-style sum of gate values
- Runs experiments for:\
  `lambda in [0.0001, 0.001, 0.01]`
- Reports test accuracy and sparsity (`% gates < 1e-2`)
- Saves:
  - `results.txt`
  - `gate_distribution.png`

## Design Decisions

- Used sigmoid gating to ensure differentiability
- Applied L1 penalty to induce sparsity in gates
- Chose a shallow architecture to allow fast experimentation
- Limited epochs due to time constraints while preserving behavior trends

## Key Insight

The gating mechanism acts as a soft mask over weights.
During training:
- Important connections → gates remain high
- Unimportant connections → gates shrink toward zero

Thus, pruning is learned, not manually applied.

## Results

| Lambda | Accuracy | Sparsity |
|--------|---------|----------|
| 0.0001 | 45.81% | 66.15% |
| 0.001  | 42.40% | 99.14% |
| 0.01   | 35.51% | 99.84% |

## Result Analysis

- At low λ (0.0001), the model retains more connections, resulting in higher accuracy but moderate sparsity.
- At medium λ (0.001), sparsity increases significantly with a slight drop in accuracy.
- At high λ (0.01), almost all connections are pruned, leading to very high sparsity but reduced accuracy.

This demonstrates a clear sparsity–accuracy trade-off controlled by λ.

## Practical Observation

With very high λ, the network tends to over-prune, collapsing most gates toward zero.
This indicates the importance of carefully tuning λ to balance model capacity and efficiency.

## Note
This implementation focuses on correctness, interpretability, and rapid experimentation rather than heavy training, aligning with practical engineering constraints.

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Expected outputs

- Console logs for each lambda with accuracy and sparsity
- `results.txt`
- `gate_distribution.png`
