from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass
class RMSConfig:
    eps: float = 1e-8
    clip: Optional[float] = 10.0
    dtype: torch.dtype = torch.float32
    device: Optional[torch.device] = None
    track: bool = True


class RunningMeanStd(torch.nn.Module):
    """Per-dimension running mean and variance"""

    def __init__(self, shape: tuple[int, ...], cfg: RMSConfig):
        super().__init__()
        assert cfg.device is not None
        self.kwargs = {"dtype": torch.float64, "device": cfg.device}
        self.register_buffer("count", torch.zeros((), **self.kwargs))
        self.register_buffer("rmean", torch.zeros(shape, **self.kwargs))
        self.register_buffer("emtwo", torch.zeros(shape, **self.kwargs))
        self.cfg = cfg

    @property
    def var(self) -> torch.Tensor:
        return self.emtwo / self.count.clamp(min=1.0)

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var + self.cfg.eps)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if not self.cfg.track:
            return
        x = x.to(dtype=torch.float64, device=self.cfg.device)
        if x.ndim == self.rmean.ndim:
            x = x.unsqueeze(0)

        b = x.size(0)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, correction=0)
        batch_count = torch.as_tensor(float(b), **self.kwargs)

        delta = batch_mean - self.rmean
        tot_count = self.count + batch_count

        new_rmean = self.rmean + delta * (batch_count / tot_count)
        new_emtwo = (
            self.emtwo +
            (batch_var * batch_count) +
            (delta.pow(2) * self.count * batch_count / tot_count)
        )

        self.rmean.copy_(new_rmean)
        self.emtwo.copy_(new_emtwo)
        self.count.copy_(tot_count)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        rmean = self.rmean.to(self.cfg.dtype)
        std = self.std.to(self.cfg.dtype)

        x = x.to(dtype=self.cfg.dtype, device=self.cfg.device)
        z = (x - rmean) / std
        if self.cfg.clip is not None:
            z = torch.clamp(z, -self.cfg.clip, self.cfg.clip)
        return z

    def state(self) -> dict[str, Union[torch.Tensor, RMSConfig]]:
        return {
            "count": self.count.detach().cpu().item(),
            "rmean": self.rmean.detach().cpu(),
            "emtwo": self.emtwo.detach().cpu(),
            "cfg": self.cfg,
        }

    def load_state(self, s: dict) -> None:
        self.count.copy_(torch.as_tensor(s["count"], **self.kwargs))
        self.rmean.copy_(s["rmean"].to(torch.float64).to(self.rmean.device))
        self.emtwo.copy_(s["emtwo"].to(torch.float64).to(self.emtwo.device))
        self.cfg = s["cfg"]
