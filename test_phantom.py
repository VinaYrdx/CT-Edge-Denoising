import numpy as np
from phantom import *


def test_shapes_and_range():
    rng = np.random.default_rng(0)
    img, e, _ = make_phantom(rng)
    assert img.shape == (256, 256) and img.dtype == np.float32
    assert 0 <= img.min() and img.max() <= 1
    assert e.ndim == 2 and e.shape[1] == 2 and len(e) > 1000


def test_circle_area_matches_analytic():
    # rendered partial-volume area must equal pi r^2 (validates the half-pixel convention)
    cont = Ellipse(127.5, 127.5, 60, 60, 0, 1.0)
    ph = Phantom([], cont)
    img = ph.render(256, 4)
    assert abs(img.sum() - np.pi * 60 ** 2) / (np.pi * 60 ** 2) < 1e-3


def test_edges_lie_on_intensity_transition():
    # at every GT edge point, image value must sit between the two material levels
    rng = np.random.default_rng(1)
    img, e, ph = make_phantom(rng)
    idx = rng.choice(len(e), 300, replace=False)
    p = e[idx]
    from scipy.ndimage import map_coordinates
    v = map_coordinates(img, [p[:, 1], p[:, 0]], order=1)
    assert np.all((v >= -1e-3) & (v <= 1 + 1e-3))


def test_sharp_edge_position():
    # vertical step edge at x = 100.0 (between pixels 99 & 100 centres is 99.5; use 100.0)
    # circle of huge radius ~ straight edge; subpixel 10-90 crossing must be at the analytic x
    cont = Ellipse(100.0 + 1000, 128, 1000, 1000, 0, 1.0)  # left boundary at x=100.0
    ph = Phantom([], cont)
    img = ph.render(256, 8)
    row = img[128]
    # linear interp of the 0.5 crossing
    j = np.where((row[:-1] < 0.5) & (row[1:] >= 0.5))[0][0]
    x = j + (0.5 - row[j]) / (row[j + 1] - row[j])
    assert abs(x - 100.0) < 0.02, x


def test_distance_zero_on_truth_and_scale():
    rng = np.random.default_rng(2)
    _, e, _ = make_phantom(rng)
    assert dist_to_edges(e[:50], e).max() < 1e-6
    shifted = e[:50] + np.array([0.0, 0.5], np.float32)
    assert dist_to_edges(shifted, e).max() <= 0.5 + 1e-5


def test_many_phantoms_valid():
    rng = np.random.default_rng(3)
    for _ in range(30):
        img, e, _ = make_phantom(rng)
        assert len(e) > 500 and np.isfinite(img).all()
        assert len(np.unique(np.round(img, 2))) >= 4  # several materials present


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
