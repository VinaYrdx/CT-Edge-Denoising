"""Step 2: scanner noise. phantom -> sinogram -> Beer-Lambert -> Poisson + Gaussian -> -log -> FBP.
Image values = normalised attenuation (1.0 = steel). mu_px = attenuation per pixel at value 1.0,
so line integral p = radon(img) * mu_px. Frame is identical to the input (edge truth stays valid)."""
from __future__ import annotations
import numpy as np
from skimage.transform import radon, iradon


def angles(n_angles: int = 360) -> np.ndarray:
    return np.linspace(0.0, 180.0, n_angles, endpoint=False)


def project(img: np.ndarray, theta: np.ndarray, mu_px: float = 0.02) -> np.ndarray:
    """Clean sinogram of line integrals p, shape [n_det, n_angles]. Image must sit inside the inscribed circle."""
    return radon(img.astype(np.float64), theta=theta, circle=True) * mu_px


def noisy_sino(p: np.ndarray, N0: float, sigma_e: float, rng: np.random.Generator) -> np.ndarray:
    """Beer-Lambert -> Poisson photon counts -> Gaussian electronic noise (in photons) -> -log."""
    counts = rng.poisson(N0 * np.exp(-p)).astype(np.float64)
    counts += rng.normal(0.0, sigma_e, p.shape)
    counts = np.maximum(counts, 1.0)  # photon starvation guard (log of <=0)
    return -np.log(counts / N0)


def reconstruct(p: np.ndarray, theta: np.ndarray, n: int, mu_px: float = 0.02, filter_name: str = "ramp") -> np.ndarray:
    """Filtered back-projection, back to image units. Not clipped (noise stays symmetric)."""
    return (iradon(p / mu_px, theta=theta, filter_name=filter_name, circle=True, output_size=n)).astype(np.float32)


def scan(img: np.ndarray, N0: float = 1e5, sigma_e: float = 5.0, n_angles: int = 360,
         mu_px: float = 0.02, rng: np.random.Generator | None = None, filter_name: str = "ramp") -> np.ndarray:
    rng = rng or np.random.default_rng()
    th = angles(n_angles)
    return reconstruct(noisy_sino(project(img, th, mu_px), N0, sigma_e, rng), th, img.shape[0], mu_px, filter_name)


# ------------------------------- tests -------------------------------
def _ph(seed=0):
    from phantom import make_phantom
    return make_phantom(np.random.default_rng(seed))[0]


def test_noise_free_roundtrip_and_no_shift():
    img, th = _ph(), angles(360)
    rec = reconstruct(project(img, th), th, 256)
    rmse = lambda dy, dx: np.sqrt(np.mean((np.roll(rec, (dy, dx), (0, 1))[8:-8, 8:-8] - img[8:-8, 8:-8]) ** 2))
    r0 = rmse(0, 0)
    assert r0 < 0.06, r0
    for s in [(1, 0), (-1, 0), (0, 1), (0, -1)]:   # zero shift must be the best alignment (frame preserved)
        assert r0 < rmse(*s), (s, r0, rmse(*s))


def test_poisson_stats_and_beer_lambert():
    rng = np.random.default_rng(1)
    p = np.full((200, 200), 1.5)
    c = rng.poisson(1e5 * np.exp(-p)).astype(float)
    m = 1e5 * np.exp(-1.5)
    assert abs(c.mean() - m) / m < 0.01 and abs(c.var() - m) / m < 0.05   # variance == mean
    s = noisy_sino(p, 1e5, 0.0, np.random.default_rng(2))
    assert abs(s.mean() - 1.5) < 0.01


def test_more_photons_less_noise():
    img, th = _ph(), angles(180)
    p = project(img, th)
    err = lambda N0: np.sqrt(np.mean((reconstruct(noisy_sino(p, N0, 5.0, np.random.default_rng(3)), th, 256) - reconstruct(p, th, 256)) ** 2))
    e_lo, e_hi = err(5e4), err(8e5)
    assert e_lo / e_hi > 2.5, (e_lo, e_hi)   # expect ~ sqrt(16)=4


def test_noise_is_correlated_not_white():
    img, th = _ph(), angles(360)
    p = project(img, th)
    n = reconstruct(noisy_sino(p, 1e5, 5.0, np.random.default_rng(4)), th, 256) - reconstruct(p, th, 256)
    n = n[64:192, 64:192]; n = n - n.mean()
    lag1 = (n[:, 1:] * n[:, :-1]).mean() / n.var()
    assert lag1 > 0.05, lag1                 # white noise ~0 (+-0.01)


def test_shape_dtype_finite_seeded_starvation():
    img = _ph()
    a = scan(img, rng=np.random.default_rng(5)); b = scan(img, rng=np.random.default_rng(5))
    assert a.shape == (256, 256) and a.dtype == np.float32 and np.isfinite(a).all() and np.array_equal(a, b)
    d = scan(img, mu_px=0.5, N0=1e4, rng=np.random.default_rng(6))   # absurdly dense: must not NaN/inf
    assert np.isfinite(d).all()


if __name__ == "__main__":
    import time
    t = time.time()
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
    print("All scanner tests passed.", round(time.time() - t, 1), "s")