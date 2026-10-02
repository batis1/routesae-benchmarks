# RouteSAE citation audit

Checked 2026-10-02. Search covered the RouteSAE conference and arXiv records,
forward-citation leads, full papers and available repositories. Citation
databases have incomplete and differently merged records; this is not proof
that no qualifying paper exists.

**No verified paper in the checked set introduces a method, directly evaluates
RouteSAE alongside a broad SAE suite, and releases runnable source code.**

| Citing method | Latest version checked | What the experiments actually compare | Code |
| --- | --- | --- | --- |
| [Binary Autoencoder](https://arxiv.org/html/2509.20997v2) | February 2026 v2 | ReLU, TopK, Gated, rescaled ReLU and transcoder; RouteSAE appears in related work | Public release not verified; paper states release upon acceptance |
| [CLVQ-VAE](https://arxiv.org/html/2506.20040v4) | September 29, 2026 v4; TMLR | Clustering, single-layer VQ-VAE, cross-layer SAE and single-layer SAE; not RouteSAE | [Official repository](https://github.com/agarg-dev/CLVQVAE) |
| [FAST](https://arxiv.org/html/2506.07691v2) | July 2026 v2 | Training paradigms for Standard/ReLU and JumpReLU SAE; RouteSAE is described as orthogonal | [Official repository](https://github.com/Geaming2002/FAST) |

The distinction matters: citing RouteSAE or improving over another cross-layer
SAE does not establish performance against RouteSAE.

## Baseline suite for the full experiment

Start with the [official RouteSAE implementation](https://github.com/swei2001/RouteSAEs)
and its exact paper configuration. Then add published TopK/BatchTopK, JumpReLU,
Gated, Matryoshka and multi-layer baselines through their official code or
maintained reference implementations. [SAEBench](https://github.com/adamkarvonen/SAEBench)
provides evaluation beyond reconstruction; suitability for multi-layer
features must be checked instead of assuming a direct adapter exists.

The current pilot tests architecture hypotheses under common reconstruction
targets. It is not yet this full baseline suite and cannot support an
"outperforms all SAEs" claim.
