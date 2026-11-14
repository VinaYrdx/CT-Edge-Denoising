from __future__ import annotations

import inspect
import itertools
import math
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np
import torch


class TunedParams(dict):
    """dict subclass holding the best parameter mapping plus .best_psnr and .scores."""

    best_psnr: float
    scores: list[tuple[dict[str, Any], float]]

    def __init__(
        self,
        params: Mapping[str, Any],
        best_psnr: float = float("nan"),
        scores: list[tuple[dict[str, Any], float]] | None = None,
    ) -> None:
        super().__init__(params)
        self.best_psnr = float(best_psnr)
        self.scores = scores if scores is not None else []


def _to_2d_numpy(img: torch.Tensor | np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Extracts a contiguous 2D (H, W) float32 numpy array from a [1, H, W] or [H, W]
    torch.Tensor or np.ndarray, storing metadata to restore container, shape, and device.
    """
    is_tensor = isinstance(img, torch.Tensor)
    device = img.device if is_tensor else None
    orig_shape = tuple(img.shape)

    if is_tensor:
        arr = img.detach().cpu().numpy()
    elif isinstance(img, np.ndarray):
        arr = img
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(img).__name__}")

    if arr.ndim == 2:
        arr_2d = np.ascontiguousarray(arr, dtype=np.float32)
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr_2d = np.ascontiguousarray(arr[0], dtype=np.float32)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr_2d = np.ascontiguousarray(arr[..., 0], dtype=np.float32)
    else:
        raise ValueError(
            f"Expected [1, H, W] or [H, W] single-channel image, got shape {orig_shape}"
        )

    meta = {"is_tensor": is_tensor, "device": device, "orig_shape": orig_shape}
    return arr_2d, meta


def _from_2d_numpy(
    arr_2d: np.ndarray, meta: dict[str, Any]
) -> torch.Tensor | np.ndarray:
    """Restores a 2D (H, W) float32 array to its original container type, shape, and device."""
    out = np.ascontiguousarray(arr_2d, dtype=np.float32).reshape(meta["orig_shape"])
    if meta["is_tensor"]:
        return torch.from_numpy(out).to(device=meta["device"], dtype=torch.float32)
    return out


def compute_psnr(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    data_range: float = 1.0,
    eps: float = 1e-12,
) -> float:
    """Computes Peak Signal-to-Noise Ratio (dB) on float32 images in [0, data_range]."""
    p_2d, _ = _to_2d_numpy(pred)
    t_2d, _ = _to_2d_numpy(target)
    if p_2d.shape != t_2d.shape:
        raise ValueError(f"Shape mismatch in compute_psnr: {p_2d.shape} vs {t_2d.shape}")
    mse = float(np.mean((p_2d - t_2d) ** 2, dtype=np.float64))
    return float(10.0 * math.log10((data_range * data_range) / max(mse, eps)))


def bilateral(
    img: torch.Tensor | np.ndarray,
    d: int = 5,
    sigma_color: float = 0.1,
    sigma_space: float = 2.0,
) -> torch.Tensor | np.ndarray:
    """
    Edge-preserving bilateral filter via cv2.bilateralFilter on float32.
    Preserves input container (torch.Tensor or np.ndarray), shape ([1, H, W] or [H, W]),
    and float32 dtype.
    """
    if sigma_color <= 0.0 or sigma_space <= 0.0:
        raise ValueError("sigma_color and sigma_space must be > 0")

    arr_2d, meta = _to_2d_numpy(img)
    if float(arr_2d.max() - arr_2d.min()) == 0.0:
        return _from_2d_numpy(arr_2d.copy(), meta)

    filtered = cv2.bilateralFilter(
        arr_2d,
        d=int(d),
        sigmaColor=float(sigma_color),
        sigmaSpace=float(sigma_space),
        borderType=cv2.BORDER_REFLECT,
    )
    return _from_2d_numpy(filtered, meta)


def perona_malik(
    img: torch.Tensor | np.ndarray,
    n_iter: int = 15,
    kappa: float = 0.05,
    gamma: float = 0.15,
) -> torch.Tensor | np.ndarray:
    """
    4-neighbour Perona-Malik anisotropic diffusion in numpy with exponential conductance:
        c(delta) = exp(-(delta / kappa)**2)
    Uses explicit forward Euler integration with zero-flux Neumann boundary conditions.
    Requires 0 < gamma <= 0.25 for 2D 4-neighbour explicit stability.
    """
    if n_iter < 0:
        raise ValueError(f"n_iter ({n_iter}) must be >= 0")
    if kappa <= 0.0:
        raise ValueError(f"kappa ({kappa}) must be > 0")
    if gamma <= 0.0 or gamma > 0.25:
        raise ValueError(f"gamma ({gamma}) must be in (0, 0.25] for 4-neighbour stability")

    arr_2d, meta = _to_2d_numpy(img)
    u = arr_2d.copy()
    inv_kappa_sq = np.float32(1.0 / (kappa * kappa))
    step = np.float32(gamma)

    for _ in range(int(n_iter)):
        dy = u[1:, :] - u[:-1, :]  # (H-1, W)
        dx = u[:, 1:] - u[:, :-1]  # (H, W-1)

        flux_y = np.exp(-(dy * dy) * inv_kappa_sq) * dy
        flux_x = np.exp(-(dx * dx) * inv_kappa_sq) * dx

        u[:-1, :] += step * flux_y
        u[1:, :] -= step * flux_y
        u[:, :-1] += step * flux_x
        u[:, 1:] -= step * flux_x

    return _from_2d_numpy(u, meta)


def _expand_param_grid(
    filter_fn: Callable[..., Any],
    param_grid: Mapping[str, Any] | Sequence[Any],
) -> list[dict[str, Any]]:
    """Normalizes a dict-of-lists or sequence of dicts/tuples into a list of kwarg dicts."""
    if isinstance(param_grid, Mapping):
        if not param_grid:
            raise ValueError("param_grid cannot be empty")
        keys = list(param_grid.keys())
        val_lists = [
            list(v) if isinstance(v, (list, tuple, np.ndarray)) else [v]
            for v in param_grid.values()
        ]
        if any(len(v) == 0 for v in val_lists):
            raise ValueError("param_grid contains an empty candidate sequence")
        return [dict(zip(keys, combo)) for combo in itertools.product(*val_lists)]

    candidates = list(param_grid)
    if not candidates:
        raise ValueError("param_grid cannot be empty")
    if all(isinstance(c, Mapping) for c in candidates):
        return [dict(c) for c in candidates]

    param_names = list(inspect.signature(filter_fn).parameters.keys())[1:]
    return [dict(zip(param_names, c)) for c in candidates]


def tune_params(
    filter_fn: Callable[..., torch.Tensor | np.ndarray],
    param_grid: Mapping[str, Any] | Sequence[Any],
    noisy_list: Sequence[torch.Tensor | np.ndarray],
    clean_list: Sequence[torch.Tensor | np.ndarray],
    return_score: bool = False,
) -> TunedParams | tuple[TunedParams, float]:
    """
    Evaluates param_grid on (noisy_list, clean_list) and selects the parameter
    combination maximizing mean per-image PSNR (dB).
    Returns TunedParams (a dict subclass usable directly as **best_params, with
    .best_psnr and .scores attributes), or (best_params, best_psnr) if return_score=True.
    """
    noisy_items = list(noisy_list)
    clean_items = list(clean_list)
    if not noisy_items or len(noisy_items) != len(clean_items):
        raise ValueError(
            f"noisy_list ({len(noisy_items)}) and clean_list ({len(clean_items)}) "
            "must be non-empty and equal length"
        )

    combos = _expand_param_grid(filter_fn, param_grid)
    best_combo: dict[str, Any] = combos[0]
    best_psnr = -float("inf")
    scores: list[tuple[dict[str, Any], float]] = []

    for params in combos:
        psnr_sum = 0.0
        for noisy_img, clean_img in zip(noisy_items, clean_items):
            denoised = filter_fn(noisy_img, **params)
            psnr_sum += compute_psnr(denoised, clean_img)
        mean_psnr = psnr_sum / len(noisy_items)
        scores.append((params, mean_psnr))
        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            best_combo = params

    result = TunedParams(best_combo, best_psnr=best_psnr, scores=scores)
    if return_score:
        return result, best_psnr
    return result


# =============================================================================
# Unit, Shape & Edge-Preservation Tests
# =============================================================================


def _make_noisy_step_edge(
    shape: tuple[int, ...] = (1, 128, 128),
    low: float = 0.2,
    high: float = 0.8,
    sigma: float = 0.08,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    clean = np.full(shape, low, dtype=np.float32)
    clean[..., shape[-1] // 2 :] = high
    noise = rng.normal(0.0, sigma, size=shape).astype(np.float32)
    noisy = np.clip(clean + noise, 0.0, 1.0).astype(np.float32)
    return torch.from_numpy(noisy), torch.from_numpy(clean)


def test_shape_and_dtype() -> None:
    """Verifies [1, H, W] and [H, W] shape/dtype preservation for both Tensor and ndarray."""
    t_3d = torch.rand((1, 64, 64), dtype=torch.float32)
    n_3d = t_3d.numpy()
    t_2d = t_3d.squeeze(0)

    arr_2d, meta = _to_2d_numpy(t_3d)
    assert arr_2d.shape == (64, 64) and arr_2d.dtype == np.float32
    restored = _from_2d_numpy(arr_2d, meta)
    assert isinstance(restored, torch.Tensor) and restored.shape == (1, 64, 64)

    for fn, kwargs in [
        (bilateral, {"d": 5, "sigma_color": 0.1, "sigma_space": 2.0}),
        (perona_malik, {"n_iter": 10, "kappa": 0.05, "gamma": 0.15}),
    ]:
        out_t3 = fn(t_3d, **kwargs)
        assert isinstance(out_t3, torch.Tensor)
        assert out_t3.shape == (1, 64, 64) and out_t3.dtype == torch.float32
        assert 0.0 <= float(out_t3.min()) and float(out_t3.max()) <= 1.0

        out_n3 = fn(n_3d, **kwargs)
        assert isinstance(out_n3, np.ndarray)
        assert out_n3.shape == (1, 64, 64) and out_n3.dtype == np.float32
        assert np.allclose(out_t3.numpy(), out_n3, atol=1e-6)

        out_t2 = fn(t_2d, **kwargs)
        assert isinstance(out_t2, torch.Tensor)
        assert out_t2.shape == (64, 64) and out_t2.dtype == torch.float32


def test_constant_image_unchanged() -> None:
    """Verifies constant images in [0, 1] remain strictly unchanged."""
    const_t = torch.full((1, 64, 64), 0.37, dtype=torch.float32)
    out_b = bilateral(const_t, d=7, sigma_color=0.1, sigma_space=3.0)
    out_pm = perona_malik(const_t, n_iter=25, kappa=0.05, gamma=0.2)

    assert torch.equal(out_b, const_t)
    assert torch.equal(out_pm, const_t)


def test_noisy_step_edge_gets_lower_mse() -> None:
    """Verifies both filters lower MSE on a noisy step edge while preserving the edge jump."""
    noisy, clean = _make_noisy_step_edge(shape=(1, 128, 128), sigma=0.08, seed=42)
    mse_in = float(torch.mean((noisy - clean) ** 2).item())

    out_b = bilateral(noisy, d=5, sigma_color=0.15, sigma_space=2.0)
    out_pm = perona_malik(noisy, n_iter=20, kappa=0.08, gamma=0.15)

    mse_b = float(torch.mean((out_b - clean) ** 2).item())
    mse_pm = float(torch.mean((out_pm - clean) ** 2).item())

    assert mse_b < 0.25 * mse_in
    assert mse_pm < 0.25 * mse_in

    clean_jump = float((clean[0, :, 64] - clean[0, :, 63]).mean().item())
    b_jump = float((out_b[0, :, 64] - out_b[0, :, 63]).mean().item())
    pm_jump = float((out_pm[0, :, 64] - out_pm[0, :, 63]).mean().item())
    assert b_jump > 0.9 * clean_jump
    assert pm_jump > 0.9 * clean_jump


def test_perona_malik_conservation_and_bounds() -> None:
    """Verifies discrete intensity conservation (Neumann zero-flux) and stability checks."""
    img = torch.rand((1, 32, 32), dtype=torch.float32)
    out = perona_malik(img, n_iter=15, kappa=0.1, gamma=0.2)
    assert math.isclose(float(img.sum().item()), float(out.sum().item()), rel_tol=1e-5)

    try:
        perona_malik(img, n_iter=5, kappa=0.1, gamma=0.3)
        raise AssertionError("Expected ValueError for unstable gamma > 0.25")
    except ValueError:
        pass


def test_compute_psnr() -> None:
    a = torch.zeros((1, 16, 16), dtype=torch.float32)
    b = torch.full((1, 16, 16), 0.1, dtype=torch.float32)  # MSE = 0.01 -> PSNR = 20.0 dB
    assert math.isclose(compute_psnr(a, b), 20.0, abs_tol=1e-5)


def test_tune_params_picks_best_psnr() -> None:
    """Verifies tune_params picks the highest-PSNR parameters across dict and list grids."""
    noisy_list, clean_list = zip(
        *[_make_noisy_step_edge(seed=100 + i) for i in range(3)]
    )
    noisy_in_psnr = float(
        np.mean([compute_psnr(n, c) for n, c in zip(noisy_list, clean_list)])
    )

    b_grid = {
        "d": [5],
        "sigma_color": [0.001, 0.15, 2.0],
        "sigma_space": [2.0],
    }
    best_b = tune_params(bilateral, b_grid, noisy_list, clean_list)
    assert isinstance(best_b, dict)
    assert best_b["sigma_color"] == 0.15
    assert best_b.best_psnr > noisy_in_psnr + 6.0
    assert best_b.best_psnr == max(score for _, score in best_b.scores)

    pm_grid = [
        {"n_iter": 20, "kappa": 0.005, "gamma": 0.15},
        {"n_iter": 20, "kappa": 0.08, "gamma": 0.15},
        {"n_iter": 20, "kappa": 5.0, "gamma": 0.15},
    ]
    best_pm, best_pm_psnr = tune_params(
        perona_malik, pm_grid, noisy_list, clean_list, return_score=True
    )
    assert best_pm["kappa"] == 0.08
    assert best_pm_psnr > noisy_in_psnr + 6.0

    denoised = bilateral(noisy_list[0], **best_b)
    assert denoised.shape == (1, 128, 128) and denoised.dtype == torch.float32


if __name__ == "__main__":
    test_shape_and_dtype()
    test_constant_image_unchanged()
    test_noisy_step_edge_gets_lower_mse()
    test_perona_malik_conservation_and_bounds()
    test_compute_psnr()
    test_tune_params_picks_best_psnr()
    print("All 6 unit, shape, and physics tests in baselines.py passed.")