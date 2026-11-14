"""Step 5: datasets. Pure numpy. Phantoms -> scanner noise -> cached .npz. Splits use disjoint seed ranges.
clean = the phantom itself (the target). noisy = FBP reconstruction of a noisy scan (N0 = photons per ray).
Note: even a noise-free FBP of a phantom is ~37 dB PSNR from the phantom, so N0 above ~5e4 all look alike."""
from __future__ import annotations
import os
import numpy as np
from phantom import Ellipse, Phantom, make_phantom, edge_distance_map, LEVELS
from scanner import angles, project, noisy_sino, reconstruct

TRAIN_N0 = (2.5e3, 5e3, 1e4, 2e4)                  # noisy PSNR ~26-33 dB
TEST_N0 = (1e3, 2.5e3, 5e3, 1e4, 2e4, 1e5)         # 1e3 (harder) and 1e5 (easier) are unseen in training
HOLE_N0 = (5e3, 2e4)
SEEDS = dict(train=range(0, 200), val=range(1000, 1020), test=range(2000, 2030), holes=range(3000, 3030))
LEVELS_FOR = dict(train=TRAIN_N0, val=TEST_N0, test=TEST_N0, holes=HOLE_N0)


def make_hole_phantom(rng, N: int = 256):
    """A part (disk) with 3-6 circular void holes of known centre/radius. Returns img, edge_pts, phantom, holes[(cx,cy,r)]."""
    c = N / 2 - 0.5
    R = rng.uniform(0.38, 0.44) * N
    cont = Ellipse(c, c, R, R, 0, rng.choice(LEVELS[:2]))
    holes, k = [], int(rng.integers(3, 7))
    for _ in range(500):
        if len(holes) >= k:
            break
        r = rng.uniform(5, 24)
        ang, rad = rng.uniform(0, 2 * np.pi), rng.uniform(0, R - r - 8)
        cx, cy = c + rad * np.cos(ang), c + rad * np.sin(ang)
        if all(np.hypot(cx - x, cy - y) > r + rr + 10 for x, y, rr in holes):
            holes.append((cx, cy, r))
    ph = Phantom([Ellipse(x, y, r, r, 0, 0.0) for x, y, r in holes], cont)
    return ph.render(N, 4), ph.edge_points(N), ph, holes


def truth(name: str, seed: int, N: int = 256):
    """Regenerate (clean, edge_pts, holes) for a seed (deterministic, so truth is never stored)."""
    rng = np.random.default_rng(seed)
    if name == "holes":
        img, e, _, holes = make_hole_phantom(rng, N)
        return img, e, holes
    img, e, _ = make_phantom(rng, N)
    return img, e, None


def truths(name: str, N: int = 256):
    return [truth(name, s, N) for s in SEEDS[name]]


def build_split(name: str, cache_dir: str = "data", N: int = 256, n_angles: int = 360, mu_px: float = 0.02,
                sigma_e: float = 5.0, levels=None, with_dist: bool | None = None) -> dict:
    """Returns dict(clean [n,N,N], noisy [n,L,N,N], seeds, levels[, dist [n,N,N] clipped at 16 px]). Cached on disk."""
    levels = tuple(levels or LEVELS_FOR[name])
    with_dist = (name == "train") if with_dist is None else with_dist
    path = os.path.join(cache_dir, f"{name}_{N}_{n_angles}_{len(levels)}.npz")
    if os.path.exists(path):
        d = dict(np.load(path))
        if np.allclose(d["levels"], levels) and (("dist" in d) == with_dist):
            return d
    os.makedirs(cache_dir, exist_ok=True)
    seeds, th = list(SEEDS[name]), angles(n_angles)
    clean = np.zeros((len(seeds), N, N), np.float32)
    noisy = np.zeros((len(seeds), len(levels), N, N), np.float32)
    dist = np.zeros_like(clean) if with_dist else None
    for i, s in enumerate(seeds):
        img, e, _ = truth(name, s, N)
        clean[i] = img
        p = project(img, th, mu_px)
        for j, N0 in enumerate(levels):
            noisy[i, j] = reconstruct(noisy_sino(p, N0, sigma_e, np.random.default_rng([s, j])), th, N, mu_px)
        if with_dist:
            dist[i] = np.minimum(edge_distance_map(e, N), 16.0)
    out = dict(clean=clean, noisy=noisy, seeds=np.array(seeds), levels=np.array(levels))
    if with_dist:
        out["dist"] = dist
    np.savez(path, **out)
    return out


# ------------------------------- tests -------------------------------
def test_hole_phantom_and_diameter_metric():
    from metrics import feature_diameter
    img, e, _, holes = make_hole_phantom(np.random.default_rng(0))
    assert len(holes) >= 3 and len(e) > 500
    for cx, cy, r in holes:
        f = feature_diameter(img, img, cx, cy, r)
        assert f["n_rays_used"] >= 170 and abs(f["error"]) < 0.15, (r, f["error"], f["n_rays_used"])


def test_splits_disjoint_and_truth_matches_clean():
    ids = [set(SEEDS[k]) for k in SEEDS]
    assert all(not (a & b) for i, a in enumerate(ids) for b in ids[i + 1:])
    img, e, _ = truth("test", 2001)
    assert np.array_equal(img, make_phantom(np.random.default_rng(2001))[0])


def test_build_split_small():
    import tempfile
    old = SEEDS["train"]
    SEEDS["train"] = range(0, 2)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            d = build_split("train", cache_dir=tmp, n_angles=60, levels=(2e3, 2e5))
            assert d["clean"].shape == (2, 256, 256) and d["noisy"].shape == (2, 2, 256, 256) and d["dist"].shape == (2, 256, 256)
            assert d["dist"].min() == 0 or d["dist"].min() < 0.5
            err = lambda j: np.sqrt(np.mean((d["noisy"][:, j] - d["clean"]) ** 2))
            assert err(0) > err(1)                                    # fewer photons -> noisier
            d2 = build_split("train", cache_dir=tmp, n_angles=60, levels=(2e3, 2e5))   # from cache
            assert np.array_equal(d["noisy"], d2["noisy"])
    finally:
        SEEDS["train"] = old


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
    print("All data tests passed.")