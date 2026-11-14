"""
Step 1: multi-material phantoms with EXACT edge ground truth.

Coordinate convention (matters, off-by-half bugs live here):
  x = column index, y = row index, pixel CENTER of pixel (r, c) is at (x=c, y=r).
  Same convention as skimage.measure.find_contours, so predicted contours and
  ground-truth edge points live in the same frame.

Ground truth = dense point cloud sampled ANALYTICALLY along every shape boundary
(spacing ~0.1 px), kept only where the material really changes across it.
Distance to truth for any point = KD-tree query. No threshold, no Canny.
"""
import numpy as np
from scipy.spatial import cKDTree

LEVELS = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32)  # polymer, Al, Ti-ish, steel (normalised mu). 0 = air
PROBE = 0.25  # px offset used to test "material changes across this boundary point"


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


class Ellipse:
    def __init__(self, cx, cy, a, b, th, level):
        self.c, self.a, self.b, self.R, self.level = np.array([cx, cy]), a, b, _rot(th), level

    def contains(self, p):
        q = (p - self.c) @ self.R  # rotate into shape frame
        return (q[:, 0] / self.a) ** 2 + (q[:, 1] / self.b) ** 2 <= 1.0

    def boundary(self, step=0.1):
        per = np.pi * (3 * (self.a + self.b) - np.sqrt((3 * self.a + self.b) * (self.a + 3 * self.b)))  # Ramanujan
        t = np.linspace(0, 2 * np.pi, int(per / step), endpoint=False)
        loc = np.stack([self.a * np.cos(t), self.b * np.sin(t)], 1)
        nrm = np.stack([np.cos(t) / self.a, np.sin(t) / self.b], 1)  # gradient of implicit fn
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        return loc @ self.R.T + self.c, nrm @ self.R.T


class Polygon:
    """Convex polygon, CCW vertices. Rotated rectangle is a special case."""

    def __init__(self, verts, level):
        self.v, self.level = np.asarray(verts, float), level

    def contains(self, p):
        ok = np.ones(len(p), bool)
        for i in range(len(self.v)):
            a, b = self.v[i], self.v[(i + 1) % len(self.v)]
            e = b - a
            ok &= (e[0] * (p[:, 1] - a[1]) - e[1] * (p[:, 0] - a[0])) >= 0  # left of every edge
        return ok

    def boundary(self, step=0.1):
        P, N = [], []
        for i in range(len(self.v)):
            a, b = self.v[i], self.v[(i + 1) % len(self.v)]
            e = b - a
            L = np.linalg.norm(e)
            n = int(L / step)
            t = (np.arange(n) / n)[:, None]
            P.append(a + t * e)
            N.append(np.tile([e[1] / L, -e[0] / L], (n, 1)))  # outward normal for CCW
        return np.vstack(P), np.vstack(N)


def rect(cx, cy, w, h, th, level):
    loc = np.array([[-w, -h], [w, -h], [w, h], [-w, h]])
    return Polygon(loc @ _rot(th).T + [cx, cy], level)


def rand_convex_poly(rng, cx, cy, r, level, k=None):
    k = k or rng.integers(3, 7)
    ang = np.sort(rng.uniform(0, 2 * np.pi, k))
    # points on an ellipse sorted by angle -> always convex, CCW
    th = rng.uniform(0, np.pi)
    loc = np.stack([r * np.cos(ang), r * rng.uniform(0.5, 1) * np.sin(ang)], 1)
    return Polygon(loc @ _rot(th).T + [cx, cy], level)


class Phantom:
    """Painter's algorithm: container (the 'part') first, inner shapes clipped to it."""

    def __init__(self, shapes, container):
        self.container, self.shapes = container, shapes

    def value_at(self, p):
        val = np.zeros(len(p), np.float32)  # air
        inside = self.container.contains(p)
        val[inside] = self.container.level
        for s in self.shapes:
            m = s.contains(p) & inside
            val[m] = s.level
        return val

    def render(self, N=256, ss=4):
        """Supersample ss x ss then box-average -> exact-area partial-volume edges."""
        g = (np.arange(N * ss) + 0.5) / ss - 0.5
        X, Y = np.meshgrid(g, g)
        v = self.value_at(np.stack([X.ravel(), Y.ravel()], 1)).reshape(N * ss, N * ss)
        return v.reshape(N, ss, N, ss).mean((1, 3)).astype(np.float32)

    def edge_points(self, N=256, step=0.1):
        """Analytic boundary points where material actually changes."""
        out = []
        for s in [self.container] + self.shapes:
            p, n = s.boundary(step)
            d = self.value_at(p + PROBE * n) - self.value_at(p - PROBE * n)
            keep = np.abs(d) > 1e-3
            out.append(p[keep])
        e = np.vstack(out)
        ok = (e[:, 0] > 1) & (e[:, 0] < N - 2) & (e[:, 1] > 1) & (e[:, 1] < N - 2)  # drop image border
        return e[ok].astype(np.float32)


def make_phantom(rng, N=256, ss=4):
    """Returns img [N,N] float32 in [0,1], edge_pts [M,2] (x,y) float32, Phantom object."""
    c = N / 2 - 0.5
    R = rng.uniform(0.36, 0.44) * N
    cont = Ellipse(c, c, R, R, 0, rng.choice(LEVELS[:2]))
    shapes = []
    for _ in range(rng.integers(3, 8)):  # 3-7 inner features -> 3-6 distinct levels typically
        r = rng.uniform(0.05, 0.22) * N
        ang, rad = rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.75) * R
        cx, cy = c + rad * np.cos(ang), c + rad * np.sin(ang)
        lv = rng.choice(np.concatenate([[0.0], LEVELS]))  # 0.0 = void/hole
        if lv == cont.level:
            lv = 0.0
        kind = rng.integers(0, 3)
        if kind == 0:
            s = Ellipse(cx, cy, r, r * rng.uniform(0.5, 1), rng.uniform(0, np.pi), lv)
        elif kind == 1:
            s = rect(cx, cy, r, r * rng.uniform(0.4, 1), rng.uniform(0, np.pi), lv)
        else:
            s = rand_convex_poly(rng, cx, cy, r, lv)
        shapes.append(s)
    ph = Phantom(shapes, cont)
    return ph.render(N, ss), ph.edge_points(N), ph


def dist_to_edges(query_xy, edge_pts):
    """Unsigned distance (px) from each query point to the true boundary."""
    return cKDTree(edge_pts).query(query_xy)[0]


def edge_distance_map(edge_pts, N=256):
    """Dense unsigned distance map (for edge-weighted loss later)."""
    yy, xx = np.mgrid[0:N, 0:N]
    return dist_to_edges(np.stack([xx.ravel(), yy.ravel()], 1), edge_pts).reshape(N, N).astype(np.float32)
