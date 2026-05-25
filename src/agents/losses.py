from beartype import beartype
from einops import rearrange

import torch
from torch import nn
from torch.nn import functional as ff


class HistogramLossGaussian(nn.Module):

    @beartype
    def __init__(self,
                 *,
                 min_value: float,
                 max_value: float,
                 num_bins: int,
                 sigma: float,
                 device: torch.device,
                 reduction: str = "none"):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.sigma = sigma
        self.device = device
        self.reduction = reduction
        self.support = torch.linspace(
            min_value, max_value, num_bins + 1, dtype=torch.float, device=self.device)
        self.sqrt_of_two = torch.sqrt(torch.tensor(2.0, device=self.device))

    @beartype
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = rearrange(logits, "b (c d) -> b c d", c=self.num_bins)
        target_probs = self.transform_to_probs(target)
        target_probs = rearrange(target_probs, "b d c -> b c d", c=self.num_bins)
        return ff.cross_entropy(logits, target_probs, reduction=self.reduction)

    @beartype
    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        operand1 = self.support - target.unsqueeze(-1)
        operand2 = self.sqrt_of_two * self.sigma
        operand = operand1 / operand2
        cdf_evals = torch.special.erf(operand)
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / (z + 1e-6).unsqueeze(-1)  # credit for "+ eps" trick: thanks Denis

    @beartype
    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        centers = (self.support[:-1] + self.support[1:]) / 2
        return torch.sum(probs * centers, dim=-1)

