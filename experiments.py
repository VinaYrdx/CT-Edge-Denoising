"""Step 7: evaluation helpers + the three 'new findings' analyses. numpy/pandas only (no torch).
den / clean: [n,H,W] arrays; tr: list of (clean, edge_pts, holes) from data.truths(name)."""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from metrics import psnr, ssim, edge_metrics, feature_diameter


def evaluate(den, clean, tr) -> pd.DataFrame:
    """One row per phantom: psnr, ssim, edge error (mean/p95/worst, px), 10-90 edge width (px), miss rate."""
    rows = []
    for d, c, (_, e, _) in zip(den, clean, tr):
        m, dc = edge_metrics(d, c, e), np.clip(d, 0, 1)
        rows.append(dict(psnr=psnr(dc, c), ssim=ssim(dc, c), edge=m["mean"], p95=m["p95"], worst=m["worst"],
                         esf=m["esf_median"], miss=m["miss_rate"]))
    return pd.DataFrame(rows)


def evaluate_holes(den, clean, tr, min_rays: int = 120) -> pd.DataFrame:
    """One row per hole: true radius, signed diameter error (px; negative = hole measured too small)."""
    rows = []
    for d, c, (_, _, holes) in zip(den, clean, tr):
        for cx, cy, r in holes:
            f = feature_diameter(d, c, cx, cy, r)
            if f["n_rays_used"] >= min_rays:
                rows.append(dict(radius=r, true_d=2 * r, err=f["error"]))
    return pd.DataFrame(rows)


def tune_filter(fn, grid: dict, noisy, clean, tr):
    """Grid search a classical filter. Returns (best_by_psnr, best_by_edge, table). Tune on VAL data only."""
    keys, rows = list(grid), []
    for combo in itertools.product(*grid.values()):
        kw = dict(zip(keys, combo))
        den = np.stack([fn(x.astype(np.float32), **kw) for x in noisy])
        df = evaluate(den, clean, tr)
        rows.append(dict(kw, psnr=df.psnr.mean(), edge=df.edge.mean(), esf=df.esf.mean()))
    t = pd.DataFrame(rows)
    pick = lambda i: {k: t.loc[i, k].item() if hasattr(t.loc[i, k], "item") else t.loc[i, k] for k in keys}
    return pick(t.psnr.idxmax()), pick(t.edge.idxmin()), t


def collect_blur(den, clean, tr, sigma: float):
    """Per-edge-point arrays (width, contrast, noise sigma) pooled over phantoms. Needs metrics return_raw with contrast/ok."""
    W, C = [], []
    for d, c, (_, e, _) in zip(den, clean, tr):
        m = edge_metrics(d, c, e, return_raw=True)
        keep = m["ok"] & np.isfinite(m["w"])
        W.append(m["w"][keep]); C.append(np.abs(m["contrast"][keep]))
    W, C = np.concatenate(W), np.concatenate(C)
    return W, C, np.full_like(W, sigma)


def blur_law(w, contrast, sigma, w0: float = 1.35) -> dict:
    """Fit log(excess width) = a + alpha*log(sigma) + beta*log(contrast); excess = sqrt(w^2 - w0^2) (w0 = clean/pixel-grid floor).
    Hypothesis (posterior-mean blur ~ edge-position uncertainty): alpha ~ +1, beta ~ -1."""
    w, contrast, sigma = map(np.asarray, (w, contrast, sigma))
    ex = np.sqrt(np.maximum(w ** 2 - w0 ** 2, 0.0))
    m = np.isfinite(ex) & (ex > 0.05) & (contrast > 0) & (sigma > 0)
    y = np.log(ex[m])
    A = np.c_[np.ones(m.sum()), np.log(sigma[m]), np.log(contrast[m])]
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    r2 = 1 - ((y - A @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return dict(alpha=float(coef[1]), beta=float(coef[2]), r2=float(r2), n=int(m.sum()))


def rank_report(df: pd.DataFrame) -> pd.DataFrame:
    """df: one row per method with columns psnr, edge (px, lower better), esf (px), optionally diam (|px|).
    Spearman rho between PSNR and each edge/size metric. If PSNR ranks methods like the metrology metrics, rho is strongly
    NEGATIVE (higher PSNR <-> lower error). Weak or positive rho = PSNR picks the wrong model."""
    from scipy.stats import spearmanr
    cols = [c for c in ("edge", "esf", "diam") if c in df]
    return pd.DataFrame({c: [spearmanr(df.psnr, df[c])[0]] for c in cols}, index=["rho(psnr, .)"])


# ------------------------------- tests -------------------------------
def test_evaluate_identity_and_blur():
    from scipy.ndimage import gaussian_filter
    import data
    tr = [data.truth("test", s) for s in (2000, 2001)]
    clean = np.stack([t[0] for t in tr])
    a = evaluate(clean, clean, tr)
    b = evaluate(np.stack([gaussian_filter(c, 1.5) for c in clean]), clean, tr)
    assert a.psnr.min() > 80 and (a.edge < 0.1).all() and a.miss.max() == 0
    assert (b.esf.values > a.esf.values + 1.0).all() and (b.psnr.values < a.psnr.values).all()


def test_holes_shrink_detected():
    import data
    from scipy.ndimage import gaussian_filter
    tr = [data.truth("holes", 3000)]
    clean = np.stack([tr[0][0]])
    h = evaluate_holes(clean, clean, tr)
    assert len(h) >= 3 and h.err.abs().max() < 0.15
    assert evaluate_holes(np.stack([gaussian_filter(clean[0], 1.0)]), clean, tr).err.abs().max() < 0.6


def test_blur_law_recovers_exponents():
    rng = np.random.default_rng(0)
    s, c = rng.uniform(0.02, 0.1, 4000), rng.choice([0.25, 0.5, 0.75, 1.0], 4000)
    ex = 30 * s ** 1.0 * c ** -1.0 * np.exp(0.05 * rng.standard_normal(4000))
    w = np.sqrt(ex ** 2 + 1.35 ** 2)
    f = blur_law(w, c, s)
    assert abs(f["alpha"] - 1) < 0.05 and abs(f["beta"] + 1) < 0.05 and f["r2"] > 0.95, f


def test_tune_filter_and_rank():
    import data
    from scipy.ndimage import gaussian_filter
    tr = [data.truth("test", s) for s in (2000, 2001)]
    clean = np.stack([t[0] for t in tr])
    noisy = clean + 0.05 * np.random.default_rng(0).standard_normal(clean.shape).astype(np.float32)
    bp, be, t = tune_filter(lambda x, sigma: gaussian_filter(x, sigma), dict(sigma=[0.5, 1.0, 3.0]), noisy, clean, tr)
    assert bp["sigma"] in (0.5, 1.0, 3.0) and len(t) == 3 and be["sigma"] <= bp["sigma"] + 2
    r = rank_report(pd.DataFrame(dict(psnr=[30, 31, 32, 33], edge=[0.4, 0.3, 0.2, 0.1], esf=[3, 2.5, 2, 1.5])))
    assert (r.values < -0.99).all()


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
    print("All experiments tests passed.")