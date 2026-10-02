"""Small candidate architectures for the RouteSAE pilot.

These are experimental hypotheses, not published implementations. The
RouteSAE baseline itself is imported unchanged from ``src.model``.
"""

import torch
from torch import nn

from src.model import RouteSAE, TopK


class GroupedTopK(TopK):
    """TopK dictionary with a learned sparse group gate."""

    def __init__(self, hidden_size, latent_size, k, groups=8, active_groups=1):
        if latent_size % groups:
            raise ValueError("latent_size must be divisible by groups")
        if k > latent_size // groups * active_groups:
            raise ValueError("k exceeds the available group features")
        super().__init__(hidden_size, latent_size, k)
        self.groups = groups
        self.active_groups = active_groups
        self.group_router = nn.Linear(hidden_size, groups, bias=False)

    def encode(self, x, infer_k=None, theta=None):
        if theta is not None:
            raise ValueError("threshold inference is not defined for GroupedTopK")
        k = self.k if infer_k is None else infer_k
        if k > self.latent_size // self.groups * self.active_groups:
            raise ValueError("infer_k exceeds the available group features")
        pre = self.pre_acts(x)
        scores = self.group_router(x - self.pre_bias)
        probabilities = scores.softmax(dim=-1)
        selected = scores.topk(self.active_groups, dim=-1).indices
        gate = torch.zeros_like(probabilities).scatter(-1, selected, 1.0)
        # Selected probabilities keep a gradient path into the group router.
        weights = probabilities * gate
        expanded = weights.repeat_interleave(self.latent_size // self.groups, dim=-1)
        available = gate.repeat_interleave(self.latent_size // self.groups, dim=-1).bool()
        values, indices = pre.masked_fill(~available, -torch.inf).topk(k, dim=-1)
        selected_weights = expanded.gather(-1, indices) * self.groups
        latents = torch.zeros_like(pre)
        latents.scatter_(-1, indices, values * selected_weights)
        return latents


class HierarchicalRouteSAE(RouteSAE):
    """RouteSAE with one active coarse group and a fine TopK dictionary."""

    def __init__(self, hidden_size, n_layers, latent_size, k, groups=8):
        super().__init__(hidden_size, n_layers, latent_size, k)
        self.sae = GroupedTopK(hidden_size, latent_size, k, groups, 1)


class SubspaceRouteSAE(RouteSAE):
    """RouteSAE with two active dictionary subspaces per token."""

    def __init__(self, hidden_size, n_layers, latent_size, k, groups=8):
        super().__init__(hidden_size, n_layers, latent_size, k)
        self.sae = GroupedTopK(hidden_size, latent_size, k, groups, 2)


class Top2RouteSAE(RouteSAE):
    """Use only the two most probable layers before the shared SAE."""

    def get_sae_input(self, x, router_weights, routing):
        if routing != "top2":
            return super().get_sae_input(x, router_weights, routing)
        values, indices = router_weights.topk(2, dim=-1)
        sparse = torch.zeros_like(router_weights).scatter(-1, indices, values)
        # Preserve RouteSAE's confidence scaling: do not renormalize to one.
        routed = (x * sparse.unsqueeze(-1)).sum(dim=2)
        return sparse, routed


class ResidualCascadeSAE(nn.Module):
    """Predict the last routed layer from the first and a sparse residual."""

    def __init__(self, hidden_size, latent_size, k):
        super().__init__()
        self.base = TopK(hidden_size, latent_size // 2, k // 2)
        self.delta = TopK(hidden_size, latent_size - latent_size // 2, k - k // 2)

    def forward(self, layer_stack):
        early = layer_stack[:, 0]
        late = layer_stack[:, -1]
        z_base, base_prediction = self.base(early)
        z_delta, delta_prediction = self.delta(late - base_prediction)
        return torch.cat((z_base, z_delta), dim=-1), base_prediction + delta_prediction


def normalize_decoders(model):
    """Apply the official TopK decoder column normalization to each SAE."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, TopK):
                weight = module.decoder.weight
                weight.div_(weight.norm(dim=0, keepdim=True).clamp_min(1e-8))
