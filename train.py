"""Step 6: training (torch). CropDataset, train(), denoise(), save/load. Loss runs OUTSIDE autocast, in float32."""
from __future__ import annotations
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from models import UNet, HourglassNoSkip, PlainCNN, count_params, match_plaincnn
from losses import make_loss


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s: int) -> None:
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def make_model(name: str):
    if name == "unet":
        return UNet(16, 4)
    if name == "hourglass":
        return HourglassNoSkip(16, 4)
    if name == "plain":                                   # width solved so params match the U-Net (+-5%)
        return match_plaincnn(count_params(UNet(16, 4)), depth=8)
    raise ValueError(name)


class CropDataset(Dataset):
    """Random crop + random flip/rot90 (exact, so edge truth stays valid) + random noise level per sample.
    data = dict(clean [n,N,N], noisy [n,L,N,N], dist [n,N,N] optional). num_workers must stay 0."""

    def __init__(self, data: dict, crop: int = 128, reps: int = 4, seed: int = 0):
        self.clean, self.noisy, self.dist = data["clean"], data["noisy"], data.get("dist")
        self.crop, self.reps, self.rng = crop, reps, np.random.default_rng(seed)

    def __len__(self):
        return len(self.clean) * self.reps

    def __getitem__(self, idx):
        i, j = idx % len(self.clean), int(self.rng.integers(self.noisy.shape[1]))
        N, c = self.clean.shape[-1], self.crop
        y, x = (int(v) for v in self.rng.integers(0, N - c + 1, 2))
        sl = (slice(y, y + c), slice(x, x + c))
        d = self.dist[i][sl] if self.dist is not None else np.zeros((c, c), np.float32)
        arrs = [self.noisy[i, j][sl], self.clean[i][sl], d]
        k, flip = int(self.rng.integers(4)), int(self.rng.integers(2))
        arrs = [np.rot90(a, k) for a in arrs]
        if flip:
            arrs = [a[:, ::-1] for a in arrs]
        return tuple(torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None] for a in arrs)


@torch.no_grad()
def denoise(model, noisy: np.ndarray, dev=None, bs: int = 4) -> np.ndarray:
    """noisy [n,H,W] numpy -> denoised [n,H,W] numpy (H,W divisible by 16)."""
    dev = dev or get_device()
    model = model.to(dev).eval()
    out = []
    for k in range(0, len(noisy), bs):
        x = torch.from_numpy(np.ascontiguousarray(noisy[k:k + bs], np.float32))[:, None].to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            y = model(x)
        out.append(y.float().cpu().numpy()[:, 0])
    return np.concatenate(out)


def val_l1(model, val: dict, dev) -> list:
    """Mean |denoised - clean| per noise level (monitoring only; not used to pick weights)."""
    return [float(np.abs(denoise(model, val["noisy"][:, j], dev) - val["clean"]).mean()) for j in range(val["noisy"].shape[1])]


def train(model_name: str, loss_name: str, data: dict, val: dict | None = None, epochs: int = 20, lr: float = 3e-4,
          bs: int = 8, seed: int = 0, crop: int = 128, reps: int = 4, loss_kw: dict | None = None, log=print):
    """Returns (model on CPU in eval mode, history). Final-epoch weights are kept (no checkpoint picking)."""
    set_seed(seed)
    dev = get_device()
    amp = dev.type == "cuda"
    model = make_model(model_name).to(dev)
    loss_fn = make_loss(loss_name, **(loss_kw or {})).to(dev)
    use_dist = loss_fn.w["edge"] > 0
    if use_dist and "dist" not in data:
        raise ValueError("edge loss needs data['dist'] (build_split('train') provides it)")
    dl = DataLoader(CropDataset(data, crop, reps, seed), batch_size=bs, shuffle=True, drop_last=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(dl))      # stepped per batch
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=amp)
    hist = []
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for noisy, clean, dist in dl:
            noisy, clean, dist = noisy.to(dev), clean.to(dev), dist.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                pred = model(noisy)
            loss = loss_fn(pred.float(), clean, dist if use_dist else None)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(noisy); n += len(noisy)
        row = dict(epoch=ep + 1, train=tot / n, secs=round(time.time() - t0, 1))
        if val is not None and ((ep + 1) % 5 == 0 or ep == epochs - 1):
            row["val_l1"] = val_l1(model, val, dev)
        hist.append(row)
        log(row)
    return model.cpu().eval(), hist


def save_run(path: str, model, hist, cfg: dict) -> None:
    torch.save(dict(state=model.state_dict(), hist=hist, cfg=cfg), path)


def load_run(path: str):
    ck = torch.load(path, map_location="cpu")
    m = make_model(ck["cfg"]["model"])
    m.load_state_dict(ck["state"])
    return m.eval(), ck


# ------------------------------- tests -------------------------------
def _fake(n=4, N=64):
    rng = np.random.default_rng(0)
    clean = np.zeros((n, N, N), np.float32); clean[:, :, N // 2:] = 0.8
    noisy = np.stack([clean + s * rng.standard_normal(clean.shape).astype(np.float32) for s in (0.05, 0.1)], 1)
    dist = np.broadcast_to(np.abs(np.arange(N) - N / 2).astype(np.float32), (n, N, N)).copy()
    return dict(clean=clean, noisy=noisy, dist=dist)


def test_dataset_geometry_consistent():
    A = np.random.default_rng(1).random((3, 64, 64)).astype(np.float32)
    ds = CropDataset(dict(clean=A, noisy=A[:, None].copy(), dist=A.copy()), crop=32, reps=3)
    for i in range(len(ds)):
        n, c, d = ds[i]
        assert n.shape == (1, 32, 32) and n.dtype == torch.float32
        assert torch.equal(n, c) and torch.equal(c, d)          # same crop/flip/rotation applied to all three


def test_train_each_model_and_denoise():
    d = _fake()
    for mn, ln in [("unet", "composite"), ("plain", "l1"), ("hourglass", "composite"), ("unet", "edgew+gdl")]:
        m, h = train(mn, ln, d, epochs=6, lr=1e-3, bs=4, crop=32, reps=4, log=lambda r: None)
        assert min(r["train"] for r in h[1:]) < h[0]["train"], (mn, ln, [r["train"] for r in h])
    out = denoise(m, d["noisy"][:, 0])
    assert out.shape == (4, 64, 64) and out.dtype == np.float32 and np.isfinite(out).all()


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
    print("All train tests passed.")