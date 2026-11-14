"""Step 3: metrics. Threshold-free edge metrics against exact phantom truth.
Frame: x = column, y = row, pixel centre at integer coords (same as phantom.py). Images are 2-D float arrays in [0,1].
Idea: along the TRUE edge normal, the 'edge' of an image is where its profile crosses the half-level between the two
neighbouring material values (taken from the CLEAN image). No Canny, no global threshold."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from skimage.metrics import structural_similarity

DT = 0.25  # profile step (px)


def psnr(a, b, data_range: float = 1.0) -> float:
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return float(10 * np.log10(data_range ** 2 / max(mse, 1e-12)))


def ssim(a, b) -> float:
    return float(structural_similarity(np.asarray(a, np.float32), np.asarray(b, np.float32), data_range=1.0))


def estimate_normals(edge_pts: np.ndarray, query_pts: np.ndarray, k: int = 21) -> np.ndarray:
    """(unit normals (x,y), straightness ratio) at query points via local PCA of the k nearest truth points.
    Sign is arbitrary. Ratio = small/large eigenvalue: ~0 on straight/gently curved edges, large at corners."""
    _, idx = cKDTree(edge_pts).query(query_pts, k=k)
    nb = edge_pts[idx] - edge_pts[idx].mean(1, keepdims=True)
    w, v = np.linalg.eigh(np.einsum("mki,mkj->mij", nb, nb))
    return v[:, :, 0], w[:, 0] / np.maximum(w[:, 1], 1e-12)  # smallest-variance direction = normal


def _profiles(img, pts, nrm, R, dt=DT):
    t = np.arange(-R, R + dt / 2, dt)
    xs, ys = pts[:, :1] + nrm[:, :1] * t, pts[:, 1:] + nrm[:, 1:] * t
    return t, map_coordinates(np.asarray(img, np.float64), [ys, xs], order=1, mode="nearest")


def _crossing(P, t, L, ref=None):
    """Sub-sample t where each profile crosses level L (nearest to ref, default 0). NaN if none / ref is NaN."""
    s = P - np.asarray(L, np.float64)[:, None]
    a, b = s[:, :-1], s[:, 1:]
    hit = (a * b <= 0) & (a != b)
    tc = np.where(hit, t[:-1] + (t[1] - t[0]) * (-a / np.where(hit, b - a, 1.0)), np.inf)
    r = np.zeros((len(P), 1)) if ref is None else np.asarray(ref, np.float64)[:, None]
    with np.errstate(invalid="ignore"):
        out = tc[np.arange(len(P)), np.argmin(np.abs(tc - r), axis=1)]
    return np.where(np.isfinite(out) & ~np.isnan(r[:, 0]), out, np.nan)


def _plateaus(Pc, t, flat_tol, min_contrast):
    lo, hi = Pc[:, t <= -2.0], Pc[:, t >= 2.0]
    ok = (lo.std(1) < flat_tol) & (hi.std(1) < flat_tol) & (np.abs(hi.mean(1) - lo.mean(1)) > min_contrast)
    return lo.mean(1), hi.mean(1), ok


def edge_metrics(denoised, clean, edge_pts, R: float = 4.0, stride: int = 5, k: int = 21,
                 flat_tol: float = 0.03, min_contrast: float = 0.1, return_raw: bool = False) -> dict:
    """Edge localisation + sharpness of `denoised` against exact truth `edge_pts` (from phantom.make_phantom).
    mean/p95/worst: |distance| (px) from true edge to the half-level crossing along the true normal.
    worst = directed Hausdorff (truth -> image); false edges away from truth are NOT counted.
    esf_*: 10-90% edge width (px): median of per-point widths, and width of the mean aligned profile."""
    ep = np.asarray(edge_pts, np.float64)
    pts = ep[::stride]
    nrm, ratio = estimate_normals(ep, pts, k)
    t, Pc = _profiles(clean, pts, nrm, R)
    flip = Pc[:, t >= 2.0].mean(1) < Pc[:, t <= -2.0].mean(1)      # orient low -> high
    nrm[flip] *= -1
    Pc[flip] = Pc[flip][:, ::-1]
    a, b, ok = _plateaus(Pc, t, flat_tol, min_contrast)
    ok &= (ratio < 0.02) & (np.diff(Pc, axis=1).min(1) > -0.02)     # drop corners and junctions (non-monotone clean profile)
    ok &= np.abs(np.nan_to_num(_crossing(Pc, t, (a + b) / 2), nan=9.0)) < 0.3   # self-check: clean image must agree with truth
    _, Pd = _profiles(denoised, pts, nrm, R)
    d = np.where(ok, _crossing(Pd, t, (a + b) / 2), np.nan)
    Q = (Pd - a[:, None]) / np.where(ok, b - a, 1.0)[:, None]
    w = _crossing(Q, t, np.full(len(Q), 0.9), ref=d) - _crossing(Q, t, np.full(len(Q), 0.1), ref=d)
    valid = ~np.isnan(d)
    ad = np.abs(d[valid])
    wm = np.nan
    if ok.any():
        qm = Q[ok].mean(0, keepdims=True)
        wm = float(_crossing(qm, t, [0.9])[0] - _crossing(qm, t, [0.1])[0])
    out = dict(n_used=int(ok.sum()), n_total=len(pts),
               mean=float(ad.mean()) if ad.size else np.nan,
               p95=float(np.percentile(ad, 95)) if ad.size else np.nan,
               worst=float(ad.max()) if ad.size else np.nan,
               miss_rate=float(1 - valid.sum() / max(ok.sum(), 1)),
               esf_median=float(np.nanmedian(w)) if np.isfinite(w).any() else np.nan, esf_meanprofile=wm)
    if return_raw:
        out.update(d=d, w=w, pts=pts, nrm=nrm, contrast=b - a, ok=ok)
    return out


def feature_diameter(denoised, clean, cx: float, cy: float, r_true: float, n_rays: int = 180, R: float = 4.0,
                     flat_tol: float = 0.03, min_contrast: float = 0.1) -> dict:
    """Measured diameter (px) of a circular feature (centre cx,cy, true radius r_true) via half-level radial crossings."""
    th = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    nrm = np.stack([np.cos(th), np.sin(th)], 1)
    pts = np.array([cx, cy]) + r_true * nrm
    t, Pc = _profiles(clean, pts, nrm, R)
    a, b, ok = _plateaus(Pc, t, flat_tol, min_contrast)
    _, Pd = _profiles(denoised, pts, nrm, R)
    d = np.where(ok, _crossing(Pd, t, (a + b) / 2), np.nan)
    n = int(np.isfinite(d).sum())
    est = 2 * r_true + 2 * float(np.nanmean(d)) if n else np.nan
    return dict(diameter=est, error=est - 2 * r_true, n_rays_used=n, d=d)


# ------------------------------- tests -------------------------------
def _straight_edge(N=256, x_edge=100.0):
    from phantom import Ellipse, Phantom
    ph = Phantom([], Ellipse(x_edge + 1000, 128, 1000, 1000, 0, 1.0))
    return ph.render(N, 8), ph.edge_points(N)


def test_psnr_ssim():
    a, b = np.zeros((16, 16)), np.full((16, 16), 0.1)
    assert abs(psnr(a, b) - 20.0) < 1e-6 and abs(ssim(b, b) - 1.0) < 1e-6


def test_normals_on_circle():
    from phantom import Ellipse
    p, n = Ellipse(100, 100, 40, 40, 0, 1.0).boundary(0.1)
    est = estimate_normals(p, p[::50])[0]
    assert np.abs((est * n[::50]).sum(1)).min() > 0.995


def test_clean_vs_clean_is_zero_error_and_pv_width():
    from phantom import make_phantom
    img, e, _ = make_phantom(np.random.default_rng(0))
    m = edge_metrics(img, img, e)
    assert m["n_used"] > 100 and m["miss_rate"] == 0.0, m
    assert m["mean"] < 0.06 and m["p95"] < 0.15, m
    assert 0.8 < m["esf_median"] < 1.7 and m["worst"] < 0.3, m   # pixel-grid floor (see notes)


def test_known_shift_and_known_blur():
    from scipy.ndimage import shift, gaussian_filter
    img, e = _straight_edge()
    m = edge_metrics(shift(img, (0, 0.7), order=1, mode="nearest"), img, e)
    assert abs(m["mean"] - 0.7) < 0.06, m          # edge moved 0.7 px
    s = 2.0
    m = edge_metrics(gaussian_filter(img, s), img, e)
    expect = 2.563 * np.sqrt(s * s + 1 / 12)        # 10-90 of Gaussian-blurred step (+ partial-volume box)
    assert abs(m["esf_median"] - expect) / expect < 0.10 and abs(m["esf_meanprofile"] - expect) / expect < 0.10, (m, expect)
    assert m["mean"] < 0.1                          # symmetric blur does not move the 50% edge


def test_more_blur_means_wider_and_feature_diameter():
    from scipy.ndimage import gaussian_filter
    from phantom import Ellipse, Phantom
    ph = Phantom([], Ellipse(127.5, 127.5, 20, 20, 0, 1.0))
    img, e = ph.render(256, 8), ph.edge_points(256)
    w = [edge_metrics(gaussian_filter(img, s), img, e)["esf_median"] for s in (0.5, 1.0, 1.5)]
    assert w[0] < w[1] < w[2], w
    f = feature_diameter(gaussian_filter(img, 1.5), img, 127.5, 127.5, 20.0)
    assert f["n_rays_used"] == 180 and abs(f["error"]) < 0.25, f
    assert abs(feature_diameter(img, img, 127.5, 127.5, 20.0)["error"]) < 0.1


def test_shrunk_feature_is_detected():
    from phantom import Ellipse, Phantom
    big = Phantom([], Ellipse(127.5, 127.5, 20, 20, 0, 1.0)).render(256, 8)
    small = Phantom([], Ellipse(127.5, 127.5, 19, 19, 0, 1.0)).render(256, 8)   # 2 px smaller diameter
    f = feature_diameter(small, big, 127.5, 127.5, 20.0)
    assert abs(f["error"] + 2.0) < 0.15, f


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
    print("All metrics tests passed.")