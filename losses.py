"""Step 4: losses. Tensors [B,1,H,W]. Everything is computed in float32 (safe under AMP).
pred = model output (input + residual), target = clean image."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def l1_loss(pred, target):
    return (pred.float() - target.float()).abs().mean()


def mse_loss(pred, target):
    return ((pred.float() - target.float()) ** 2).mean()


def _grads(x):
    return x[..., :, 1:] - x[..., :, :-1], x[..., 1:, :] - x[..., :-1, :]   # d/dx (columns), d/dy (rows)


def gdl_loss(pred, target, signed: bool = False):
    """Gradient-difference loss. Default = Mathieu et al.: match |gradient| magnitudes (ignores edge polarity).
    signed=True matches the signed gradients (stricter: also penalises a flipped edge)."""
    pdx, pdy = _grads(pred.float())
    tdx, tdy = _grads(target.float())
    if not signed:
        pdx, pdy, tdx, tdy = pdx.abs(), pdy.abs(), tdx.abs(), tdy.abs()
    return (pdx - tdx).abs().mean() + (pdy - tdy).abs().mean()


def _gauss_window(size: int = 11, sigma: float = 1.5, device=None):
    x = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]                              # [1,1,size,size]


def ssim(pred, target, data_range: float = 1.0, size: int = 11, sigma: float = 1.5):
    """Mean SSIM (Gaussian window, 'valid' region). Matches skimage with gaussian_weights=True, sigma=1.5,
    use_sample_covariance=False."""
    p, t = pred.float(), target.float()
    if min(p.shape[-2:]) < size:
        raise ValueError(f"image smaller than SSIM window ({size})")
    w = _gauss_window(size, sigma, p.device)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mp, mt = F.conv2d(p, w), F.conv2d(t, w)
    spp = F.conv2d(p * p, w) - mp ** 2
    stt = F.conv2d(t * t, w) - mt ** 2
    spt = F.conv2d(p * t, w) - mp * mt
    s = ((2 * mp * mt + c1) * (2 * spt + c2)) / ((mp ** 2 + mt ** 2 + c1) * (spp + stt + c2))
    return s.mean()


def ssim_loss(pred, target):
    return 1.0 - ssim(pred, target)


def edge_weight(dist_map, sigma: float = 2.0, alpha: float = 4.0):
    """1 far from edges, 1+alpha on the true edge. dist_map = distance (px) to the true edge, phantom.edge_distance_map."""
    return 1.0 + alpha * torch.exp(-(dist_map.float() ** 2) / (2 * sigma ** 2))


def edge_weighted_l1(pred, target, dist_map, sigma: float = 2.0, alpha: float = 4.0):
    """L1 weighted by closeness to the TRUE edge, normalised by the weight sum (same scale as plain L1)."""
    if dist_map.shape != pred.shape:
        raise ValueError(f"dist_map {tuple(dist_map.shape)} must match pred {tuple(pred.shape)} (crop it like the image)")
    w = edge_weight(dist_map, sigma, alpha)
    return (w * (pred.float() - target.float()).abs()).sum() / w.sum()


class CompositeLoss(nn.Module):
    """total = w_l1*L1 + w_mse*MSE + w_gdl*GDL + w_ssim*(1-SSIM) + w_edge*edgeweighted-L1. Terms with weight 0 are skipped.
    Call loss(pred, target) or loss(pred, target, dist_map). Per-term values of the last call are in .last (detached)."""

    def __init__(self, w_l1=1.0, w_mse=0.0, w_gdl=0.5, w_ssim=0.2, w_edge=0.0,
                 signed_gdl: bool = False, sigma: float = 2.0, alpha: float = 4.0):
        super().__init__()
        self.w = dict(l1=w_l1, mse=w_mse, gdl=w_gdl, ssim=w_ssim, edge=w_edge)
        if not any(self.w.values()):
            raise ValueError("all loss weights are zero")
        self.signed_gdl, self.sigma, self.alpha = signed_gdl, sigma, alpha
        self.last: dict = {}

    def forward(self, pred, target, dist_map=None):
        parts = {}
        if self.w["l1"]:
            parts["l1"] = l1_loss(pred, target)
        if self.w["mse"]:
            parts["mse"] = mse_loss(pred, target)
        if self.w["gdl"]:
            parts["gdl"] = gdl_loss(pred, target, self.signed_gdl)
        if self.w["ssim"]:
            parts["ssim"] = ssim_loss(pred, target)
        if self.w["edge"]:
            if dist_map is None:
                raise ValueError("w_edge > 0 needs dist_map")
            parts["edge"] = edge_weighted_l1(pred, target, dist_map, self.sigma, self.alpha)
        self.last = {k: v.detach() for k, v in parts.items()}
        return sum(self.w[k] * v for k, v in parts.items())


PRESETS = {                                    # names for the ablation table
    "mse": dict(w_l1=0, w_mse=1, w_gdl=0, w_ssim=0),
    "l1": dict(w_l1=1, w_gdl=0, w_ssim=0),
    "l1+gdl": dict(w_l1=1, w_gdl=0.5, w_ssim=0),
    "composite": dict(w_l1=1, w_gdl=0.5, w_ssim=0.2),
    "edgew": dict(w_l1=0, w_edge=1, w_gdl=0, w_ssim=0),
    "edgew+gdl": dict(w_l1=0, w_edge=1, w_gdl=0.5, w_ssim=0),
}


def make_loss(name: str, **overrides) -> CompositeLoss:
    """make_loss('composite', w_gdl=1.0) for the lambda sweep."""
    return CompositeLoss(**{**PRESETS[name], **overrides})


# ------------------------------- tests -------------------------------
def _step(n=64):
    x = torch.linspace(-4, 4, n)
    return (x > 0).float()[None, None, None, :].expand(1, 1, n, n).contiguous()   # vertical step edge


def test_l1_mse_known():
    a, b = torch.zeros(1, 1, 4, 4), torch.full((1, 1, 4, 4), 0.5)
    assert abs(l1_loss(a, b).item() - 0.5) < 1e-6 and abs(mse_loss(a, b).item() - 0.25) < 1e-6


def test_gdl():
    t = _step()
    assert gdl_loss(t, t).item() == 0
    assert gdl_loss(t + 0.3, t).item() < 1e-6                              # DC offset: gradients unchanged
    assert gdl_loss(F.avg_pool2d(t, 5, 1, 2), t).item() > 0.01             # blurred edge is penalised
    flipped = 1 - t
    assert gdl_loss(flipped, t).item() < 1e-6                              # |gradient| form ignores polarity
    assert gdl_loss(flipped, t, signed=True).item() > 0.01                 # signed form does not


def test_ssim_matches_skimage():
    import numpy as np
    from skimage.metrics import structural_similarity as sk
    rng = np.random.default_rng(0)
    a = rng.random((64, 64)); b = np.clip(a + 0.1 * rng.standard_normal((64, 64)), 0, 1)
    ref = sk(a, b, data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    got = ssim(torch.tensor(a)[None, None], torch.tensor(b)[None, None]).item()
    assert abs(got - ref) < 5e-4, (got, ref)
    t = torch.tensor(a)[None, None]
    assert abs(ssim(t, t).item() - 1.0) < 1e-5


def test_ssim_loss():
    t = _step()
    assert ssim_loss(t, t).item() < 1e-5 and ssim_loss(t + 0.2 * torch.randn_like(t), t).item() > 0.01


def test_edge_weighted():
    H = 32
    d = torch.full((1, 1, H, H), 50.0); d[..., :4, :] = 0.0                # 'edge' band = top 4 rows
    t = torch.zeros(1, 1, H, H)
    e_band = t.clone(); e_band[..., :4, :] = 0.5
    e_far = t.clone(); e_far[..., -4:, :] = 0.5
    assert edge_weighted_l1(e_band, t, d).item() > 2 * edge_weighted_l1(e_far, t, d).item()
    assert abs(edge_weighted_l1(e_band, t, d, alpha=0.0).item() - l1_loss(e_band, t).item()) < 1e-7


def test_composite():
    torch.manual_seed(0)
    pred = torch.rand(2, 1, 32, 32, requires_grad=True); tgt = torch.rand(2, 1, 32, 32)
    out = CompositeLoss()(pred, tgt)
    out.backward()
    assert out.ndim == 0 and torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
    expect = l1_loss(pred, tgt) + 0.5 * gdl_loss(pred, tgt) + 0.2 * ssim_loss(pred, tgt)
    assert abs(out.item() - expect.item()) < 1e-5
    assert abs(make_loss("l1")(pred, tgt).item() - l1_loss(pred, tgt).item()) < 1e-7
    assert make_loss("composite")(pred.detach().half(), tgt.half()).dtype == torch.float32
    try:
        make_loss("edgew")(pred, tgt)
        raise AssertionError("expected ValueError (no dist_map)")
    except ValueError:
        pass


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
    print("All losses tests passed.")