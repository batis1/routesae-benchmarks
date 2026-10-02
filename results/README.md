# GPU pilot results

Completed on October 2, 2026. Kaggle used seed 42 and Colab seed 43,
both on a Tesla T4. See the JSON files for exact configurations and metrics.
Qwen2.5-0.5B, WikiText-2, 8192 training tokens, 2048 validation tokens,
3584 latents, K=32, 100 optimizer steps. These are short exploratory runs.

| Comparison | Seed 42 NMSE | Seed 43 NMSE |
| --- | ---: | ---: |
| Frozen-route flat TopK | 0.4650 | 0.4510 |
| Frozen-route hierarchy | 0.5528 | 0.5437 |
| Frozen-route subspace | 0.5249 | 0.5166 |
| Common late target: top-1 routing | 0.4648 | 0.4900 |
| Common late target: top-2 routing | 0.4450 | 0.4595 |
| Common late target: direct flat TopK | 0.4315 | 0.4320 |
| Common late target: residual cascade | 0.4414 | 0.4434 |

Top-2 lowers NMSE relative to top-1 by 4.26% and 6.23%, respectively.
However, direct late-layer TopK has lower NMSE than either routing variant
on the same late-layer target. Hierarchy, subspace and cascade are worse than
their respective flat baselines in both seeds. More training may change this.

The official RouteSAE reference reconstructs a different learned target:
its NMSE is 0.4669 and 0.4533. Do not rank it against late-target values.

These runs do not establish interpretability or superiority over RouteSAE.
The strong published baseline suite, original paper protocol, downstream
KL and causal evaluations remain unfinished. Seeds also used different
PyTorch versions, so this is not a controlled cross-platform reproducibility
test. GPU memory numbers exclude activation-extraction peak requirements.

## Next experiment

Keep the direct late-layer TopK baseline in every common-target comparison.
Increase training tokens and steps, use at least three seeds per environment,
pin dependencies and model/data revisions, and compare sparsity curves.
Add official BatchTopK, JumpReLU, Gated and Matryoshka implementations with
validation-only hyperparameter selection before evaluating held-out data.
Only then test downstream loss recovery and feature interpretability.
