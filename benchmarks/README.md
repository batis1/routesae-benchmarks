# RouteSAE idea pilot

This repository preserves the official RouteSAE source at commit
`d1a3c06415c00619b3cf48895ed2a569d8b0ac5f` and adds experimental code in
`benchmarks/`. The four candidates are hypotheses. Their proposed performance
numbers have not been demonstrated.

## Run

```bash
pip install -r benchmarks/requirements.txt
python -m benchmarks.pilot --seed 42 --output results/pilot-seed42.json
```

The default pilot uses Qwen2.5-0.5B, WikiText-2 train/validation splits,
8,192 training tokens, 2,048 validation tokens, dictionary width 4 times the
residual dimension, K=32, and 100 Adam steps per model. This is a feasibility
experiment, not a reproduction of the paper's Llama/OpenWebText2 experiment.

## Comparisons

| Track | Models | Shared reconstruction target |
| --- | --- | --- |
| Official reference | Official RouteSAE hard | Its learned routed input |
| Dictionary idea | Flat TopK, one-group hierarchy, two-group subspace | The same frozen official RouteSAE routed activations |
| Routing idea | Top-1, top-2 | The last layer in the routed layer interval |
| Residual idea | Flat TopK, residual cascade | The same last-layer activations |

The routing track uses a common late-layer objective, which differs from
official RouteSAE training. Its result must be described as a controlled
routing experiment. The dictionary and cascade tracks match total latent
width and K; parameter counts are reported because gates and biases differ.
The hierarchy is a simple coarse-group prototype, not a published Tree SAE.

Lower validation NMSE is better **within each track**. NMSE values across
tracks have different targets and should not be ranked. Alive-feature counts
measure dictionary use on this small validation set; they do not measure
monosemanticity. The pilot does not compute downstream KL, human/LLM feature
interpretability, or causal faithfulness.

Before a superiority claim, run the original Llama-3.2-1B-Instruct protocol,
at least three seeds, matched sparsity/width/token budgets, validation-only
tuning, and held-out downstream interventions. Include strong published
baselines such as BatchTopK, JumpReLU, Gated SAE, Matryoshka SAE and relevant
multi-layer methods with their own official implementations. A citing paper
is not necessarily a paper that benchmarks RouteSAE.

## Colab and Kaggle

```python
!git clone https://github.com/batis1/routesae-benchmarks.git
%cd routesae-benchmarks
!pip -q install -r benchmarks/requirements.txt
!python -m benchmarks.pilot --seed 42 --output results/pilot-seed42.json
```

Use seed 42 on Kaggle and seed 43 on Colab. Both are preliminary runs.
The activation cache can be reused between seeds on the same machine.
Results include hardware, parameter counts, elapsed training time and peak
allocated GPU memory. Activation caches are excluded from Git.
