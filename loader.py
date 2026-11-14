from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import inspect
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from skimage.transform import resize
import torch
from torch.utils.data import Dataset


@dataclass
class SliceMetadata:
    slice_idx: int
    file_name: str
    z_mm: float
    proj_coord_mm: float
    dz_from_prev_mm: float | None
    dz_anomaly: bool
    orig_shape: tuple[int, int]
    out_shape: tuple[int, int]
    orig_pixel_spacing_mm: tuple[float, float]
    out_pixel_spacing_mm: tuple[float, float]
    slice_thickness_mm: float
    rescale_slope: float
    rescale_intercept: float
    non_air_fraction: float
    kvp: float | None
    exposure_mas: float | None
    reconstruction_kernel: str


def compute_slice_axis_coord(ds: Any) -> float:
    ipp = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is not None and len(iop) == 6:
        r = np.asarray(iop[:3], dtype=np.float64)
        c = np.asarray(iop[3:], dtype=np.float64)
        normal = np.cross(r, c)
        norm_len = float(np.linalg.norm(normal))
        if norm_len > 1e-6:
            normal /= norm_len
            if normal[2] < 0:
                normal = -normal
            return float(np.dot(ipp, normal))
    return float(ipp[2])


def _is_valid_axial_ct_header(ds: Any) -> bool:
    if getattr(ds, "Modality", None) != "CT":
        return False
    if not hasattr(ds, "ImagePositionPatient") or len(ds.ImagePositionPatient) != 3:
        return False
    if not hasattr(ds, "Rows") or not hasattr(ds, "Columns"):
        return False

    img_type = getattr(ds, "ImageType", [])
    if isinstance(img_type, (str, bytes)):
        img_type = [str(img_type)]
    img_type_upper = {str(t).upper() for t in img_type}
    if img_type_upper & {"LOCALIZER", "SCOUT"}:
        return False

    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is not None and len(iop) == 6:
        r = np.asarray(iop[:3], dtype=np.float64)
        c = np.asarray(iop[3:], dtype=np.float64)
        normal = np.cross(r, c)
        if abs(float(normal[2])) < 0.85:
            return False

    return True


def _discover_patient_series_headers(
    in_root: str | Path,
    z_dedup_tol_mm: float = 1e-3,
) -> dict[str, list[tuple[Path, pydicom.Dataset]]]:
    in_root = Path(in_root)
    by_patient: dict[str, dict[str, list[tuple[Path, pydicom.Dataset]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for p in sorted(in_root.rglob("*")):
        if not p.is_file():
            continue
        try:
            hdr = pydicom.dcmread(p, stop_before_pixels=True, force=False)
        except Exception:
            continue
        if _is_valid_axial_ct_header(hdr):
            pid = str(getattr(hdr, "PatientID", "")).strip() or p.parent.name or in_root.name
            uid = str(getattr(hdr, "SeriesInstanceUID", "default")).strip() or "default"
            by_patient[pid][uid].append((p, hdr))

    if not by_patient:
        raise ValueError(f"No valid axial CT DICOM slices found in {in_root}")

    winning_by_patient: dict[str, list[tuple[Path, pydicom.Dataset]]] = {}
    for pid in sorted(by_patient.keys()):
        series_map = by_patient[pid]
        best_deduped: list[tuple[Path, pydicom.Dataset]] = []
        best_key: tuple[int, int, str] = (-1, -1, "")

        for uid, pairs in series_map.items():
            modal_shape = Counter(
                (int(h.Rows), int(h.Columns)) for _, h in pairs
            ).most_common(1)[0][0]
            shaped = [
                (p, h)
                for p, h in pairs
                if (int(h.Rows), int(h.Columns)) == modal_shape
            ]
            shaped.sort(key=lambda item: compute_slice_axis_coord(item[1]))

            deduped: list[tuple[Path, pydicom.Dataset]] = []
            last_coord: float | None = None
            for p, h in shaped:
                coord = compute_slice_axis_coord(h)
                if last_coord is None or (coord - last_coord) > z_dedup_tol_mm:
                    deduped.append((p, h))
                    last_coord = coord

            cand_key = (len(deduped), len(shaped), uid)
            if cand_key > best_key:
                best_key = cand_key
                best_deduped = deduped

        if best_deduped:
            winning_by_patient[pid] = best_deduped

    if not winning_by_patient:
        raise ValueError(f"No valid axial CT series found after filtering in {in_root}")

    return winning_by_patient


def load_sorted_ct_slices(
    series_dir: str | Path,
    z_dedup_tol_mm: float = 1e-3,
    patient_id: str | None = None,
) -> list[pydicom.Dataset]:
    winning_by_patient = _discover_patient_series_headers(
        series_dir, z_dedup_tol_mm=z_dedup_tol_mm
    )

    if patient_id is not None and patient_id in winning_by_patient:
        selected_pairs = winning_by_patient[patient_id]
    elif len(winning_by_patient) == 1:
        selected_pairs = next(iter(winning_by_patient.values()))
    elif patient_id is None:
        selected_pairs = max(winning_by_patient.values(), key=len)
    else:
        raise ValueError(f"PatientID {patient_id!r} not found in {series_dir}")

    unique_slices: list[pydicom.Dataset] = []
    for p, _ in selected_pairs:
        full_ds = pydicom.dcmread(p, stop_before_pixels=False, force=False)
        if hasattr(full_ds, "PixelData"):
            unique_slices.append(full_ds)

    if not unique_slices:
        raise ValueError(f"No readable CT PixelData found in {series_dir}")

    return unique_slices


def to_hu(ds: Any, air_hu_fill: float = -1000.0) -> np.ndarray:
    raw = ds.pixel_array.astype(np.float32)

    pad_mask = np.zeros(raw.shape, dtype=bool)
    pad_val = getattr(ds, "PixelPaddingValue", None)
    if pad_val is not None:
        try:
            pv = float(pad_val)
            pad_limit = getattr(ds, "PixelPaddingRangeLimit", None)
            if pad_limit is not None:
                pl = float(pad_limit)
                lo, hi = min(pv, pl), max(pv, pl)
                pad_mask |= (raw >= lo) & (raw <= hi)
            else:
                pad_mask |= raw == pv
        except (TypeError, ValueError):
            pass

    pixel_rep = int(getattr(ds, "PixelRepresentation", 1))
    bits_stored = int(getattr(ds, "BitsStored", 16))
    if pixel_rep == 0 and bits_stored <= 12:
        pad_mask |= raw > float((1 << bits_stored) - 1)
    elif pixel_rep == 0 and bits_stored == 16:
        pad_mask |= raw >= 60000.0

    photo = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).strip().upper()
    if photo == "MONOCHROME1":
        valid_vals = raw[~pad_mask] if not np.all(pad_mask) else raw
        raw = float(np.max(valid_vals)) - raw

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    if not np.isfinite(slope) or abs(slope) < 1e-8:
        slope = 1.0
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    if not np.isfinite(intercept):
        intercept = 0.0

    hu = raw * slope + intercept
    pad_mask |= hu <= -1950.0
    if np.any(pad_mask):
        hu[pad_mask] = air_hu_fill

    return np.nan_to_num(
        hu, nan=air_hu_fill, posinf=3000.0, neginf=air_hu_fill
    ).astype(np.float32)


def has_sufficient_non_air(
    hu_img: np.ndarray,
    air_hu_thresh: float = -900.0,
    min_fraction: float = 0.05,
) -> bool:
    non_air_frac = float(np.mean(hu_img > air_hu_thresh))
    return bool(non_air_frac + 1e-6 >= min_fraction)


def window_and_normalize(
    hu_img: np.ndarray,
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
) -> np.ndarray:
    if hu_max <= hu_min:
        raise ValueError(f"hu_max ({hu_max}) must exceed hu_min ({hu_min})")
    clipped = np.clip(hu_img, hu_min, hu_max)
    norm = (clipped - hu_min) / (hu_max - hu_min)
    return norm.astype(np.float32, copy=False)


def resize_slice(
    norm_img: np.ndarray,
    out_shape: tuple[int, int] = (256, 256),
) -> np.ndarray:
    if norm_img.shape == out_shape:
        return np.clip(norm_img, 0.0, 1.0).astype(np.float32, copy=False)
    resized = resize(
        norm_img,
        out_shape,
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    )
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def _process_single_patient_pairs(
    path_hdr_pairs: list[tuple[Path, pydicom.Dataset]],
    out_root: str | Path,
    patient_id: str,
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
    air_hu_thresh: float = -900.0,
    min_non_air_frac: float = 0.05,
    out_shape: tuple[int, int] = (256, 256),
    max_slices: int | None = None,
    save_manifest: bool = True,
) -> list[Path]:
    if max_slices is not None and max_slices <= 0:
        raise ValueError(f"max_slices ({max_slices}) must be >= 1")

    safe_pid = (
        "".join(c if (c.isalnum() or c in "-_.") else "_" for c in patient_id.strip())
        or "UNKNOWN_PATIENT"
    )
    patient_out_dir = Path(out_root) / safe_pid
    patient_out_dir.mkdir(parents=True, exist_ok=True)

    coords = [compute_slice_axis_coord(h) for _, h in path_hdr_pairs]
    diffs = np.diff(coords)
    median_dz = float(np.median(diffs)) if len(diffs) > 0 else 0.0

    non_air_survivors: list[tuple[Path, float]] = []
    num_readable_raw = 0
    for p, _ in path_hdr_pairs:
        ds = pydicom.dcmread(p, stop_before_pixels=False, force=False)
        if not hasattr(ds, "PixelData"):
            continue
        num_readable_raw += 1
        hu = to_hu(ds)
        non_air_frac = float(np.mean(hu > air_hu_thresh))
        if has_sufficient_non_air(
            hu, air_hu_thresh=air_hu_thresh, min_fraction=min_non_air_frac
        ):
            non_air_survivors.append((p, non_air_frac))

    if max_slices is not None and len(non_air_survivors) > max_slices:
        idx = np.linspace(0, len(non_air_survivors) - 1, num=max_slices, dtype=int)
        non_air_survivors = [non_air_survivors[i] for i in idx]

    saved_paths: list[Path] = []
    manifest_records: list[dict[str, Any]] = []
    prev_kept_coord: float | None = None

    for save_idx, (p, non_air_frac) in enumerate(non_air_survivors):
        ds = pydicom.dcmread(p, stop_before_pixels=False, force=False)
        hu = to_hu(ds)
        norm = window_and_normalize(hu, hu_min=hu_min, hu_max=hu_max)
        resized = resize_slice(norm, out_shape=out_shape)

        fname = f"{save_idx:04d}.npy"
        out_path = patient_out_dir / fname
        np.save(out_path, resized)
        saved_paths.append(out_path)

        orig_h, orig_w = hu.shape
        ps = getattr(ds, "PixelSpacing", [1.0, 1.0])
        ps_y, ps_x = float(ps[0]), float(ps[1])
        out_ps_y = ps_y * (orig_h / float(out_shape[0]))
        out_ps_x = ps_x * (orig_w / float(out_shape[1]))

        coord = compute_slice_axis_coord(ds)
        dz = (coord - prev_kept_coord) if prev_kept_coord is not None else None
        dz_anomaly = bool(
            dz is not None and median_dz > 0 and abs(dz - median_dz) > 0.5 * median_dz
        )
        prev_kept_coord = coord

        meta = SliceMetadata(
            slice_idx=save_idx,
            file_name=fname,
            z_mm=float(ds.ImagePositionPatient[2]),
            proj_coord_mm=coord,
            dz_from_prev_mm=dz,
            dz_anomaly=dz_anomaly,
            orig_shape=(orig_h, orig_w),
            out_shape=out_shape,
            orig_pixel_spacing_mm=(ps_y, ps_x),
            out_pixel_spacing_mm=(out_ps_y, out_ps_x),
            slice_thickness_mm=float(getattr(ds, "SliceThickness", median_dz or 1.0)),
            rescale_slope=float(getattr(ds, "RescaleSlope", 1.0)),
            rescale_intercept=float(getattr(ds, "RescaleIntercept", 0.0)),
            non_air_fraction=non_air_frac,
            kvp=float(getattr(ds, "KVP", 0.0)) if hasattr(ds, "KVP") else None,
            exposure_mas=(
                float(getattr(ds, "Exposure", 0.0)) if hasattr(ds, "Exposure") else None
            ),
            reconstruction_kernel=str(getattr(ds, "ConvolutionKernel", "UNKNOWN")),
        )
        manifest_records.append(asdict(meta))

    if save_manifest:
        manifest_payload = {
            "patient_id": safe_pid,
            "num_raw_slices": num_readable_raw,
            "num_kept_slices": len(saved_paths),
            "nominal_dz_mm": median_dz,
            "hu_window": [hu_min, hu_max],
            "air_hu_thresh": air_hu_thresh,
            "min_non_air_frac": min_non_air_frac,
            "max_slices": max_slices,
            "slices": manifest_records,
        }
        with open(patient_out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2)

    return saved_paths


def process_patient_series(
    series_dir: str | Path,
    out_root: str | Path,
    patient_id: str | None = None,
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
    air_hu_thresh: float = -900.0,
    min_non_air_frac: float = 0.05,
    out_shape: tuple[int, int] = (256, 256),
    max_slices: int | None = None,
    save_manifest: bool = True,
) -> list[Path]:
    winning_by_patient = _discover_patient_series_headers(series_dir)
    if patient_id is not None and patient_id in winning_by_patient:
        disc_pid = patient_id
        pairs = winning_by_patient[patient_id]
    elif len(winning_by_patient) == 1:
        disc_pid, pairs = next(iter(winning_by_patient.items()))
    elif patient_id is None:
        disc_pid, pairs = max(winning_by_patient.items(), key=lambda kv: len(kv[1]))
    else:
        raise ValueError(f"PatientID {patient_id!r} not found in {series_dir}")

    out_pid = patient_id or disc_pid
    return _process_single_patient_pairs(
        path_hdr_pairs=pairs,
        out_root=out_root,
        patient_id=out_pid,
        hu_min=hu_min,
        hu_max=hu_max,
        air_hu_thresh=air_hu_thresh,
        min_non_air_frac=min_non_air_frac,
        out_shape=out_shape,
        max_slices=max_slices,
        save_manifest=save_manifest,
    )


def process_folder(
    in_root: str | Path,
    out_root: str | Path,
    max_slices: int = 40,
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
    air_hu_thresh: float = -900.0,
    min_non_air_frac: float = 0.05,
    out_shape: tuple[int, int] = (256, 256),
    save_manifest: bool = True,
) -> dict[str, list[Path]]:
    winning_by_patient = _discover_patient_series_headers(in_root)
    results: dict[str, list[Path]] = {}

    for pid, pairs in winning_by_patient.items():
        results[pid] = _process_single_patient_pairs(
            path_hdr_pairs=pairs,
            out_root=out_root,
            patient_id=pid,
            hu_min=hu_min,
            hu_max=hu_max,
            air_hu_thresh=air_hu_thresh,
            min_non_air_frac=min_non_air_frac,
            out_shape=out_shape,
            max_slices=max_slices,
            save_manifest=save_manifest,
        )

    return results


class LIDCSliceDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        patient_ids: list[str] | None = None,
        crop_size: int | None = None,
        random_crop: bool = False,
        return_metadata: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.return_metadata = return_metadata

        self.samples: list[tuple[Path, dict[str, Any]]] = []
        patient_dirs = sorted(p for p in self.root_dir.iterdir() if p.is_dir())
        if patient_ids is not None:
            allowed = set(patient_ids)
            patient_dirs = [p for p in patient_dirs if p.name in allowed]

        for pdir in patient_dirs:
            manifest_path = pdir / "manifest.json"
            meta_by_file: dict[str, dict[str, Any]] = {}
            if manifest_path.is_file():
                with open(manifest_path, "r", encoding="utf-8") as f:
                    mdata = json.load(f)
                for s in mdata.get("slices", []):
                    meta_by_file[s["file_name"]] = s

            for npy_path in sorted(pdir.glob("*.npy")):
                meta = meta_by_file.get(
                    npy_path.name,
                    {
                        "patient_id": pdir.name,
                        "file_name": npy_path.name,
                        "out_pixel_spacing_mm": [1.0, 1.0],
                    },
                )
                meta["patient_id"] = pdir.name
                self.samples.append((npy_path, meta))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        npy_path, meta = self.samples[idx]
        arr = np.load(npy_path, mmap_mode="r")
        img = np.array(arr, dtype=np.float32, copy=True)

        if self.crop_size is not None:
            h, w = img.shape
            ch = cw = self.crop_size
            if h < ch or w < cw:
                raise ValueError(
                    f"Slice shape {img.shape} smaller than crop_size {self.crop_size}"
                )
            if self.random_crop:
                top = int(torch.randint(0, h - ch + 1, (1,)).item())
                left = int(torch.randint(0, w - cw + 1, (1,)).item())
            else:
                top = (h - ch) // 2
                left = (w - cw) // 2
            img = img[top : top + ch, left : left + cw]

        tensor = torch.from_numpy(img).unsqueeze(0)
        if self.return_metadata:
            return tensor, meta
        return tensor


# =============================================================================
# TESTS (10)
# =============================================================================

def _make_synthetic_dicom(
    path: Path,
    raw_pixels: np.ndarray,
    z_pos: float,
    slope: float = 2.0,
    intercept: float = -1024.0,
    patient_id: str = "LIDC-IDRI-TEST",
    series_uid: str = "1.2.826.0.1.3680043.2.1125.1",
    modality: str = "CT",
    image_type: list[str] | None = None,
    iop: list[float] | None = None,
    pixel_spacing: list[float] | None = None,
    pixel_padding_value: int | None = None,
    pixel_rep: int = 1,
    bits_stored: int = 16,
    photometric: str = "MONOCHROME2",
) -> None:
    dtype = np.int16 if pixel_rep == 1 else np.uint16
    raw_bytes = np.ascontiguousarray(raw_pixels, dtype=dtype).tobytes()

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # We leave these unchanged for the test runner environment.
    # The actual loading handles proper versions implicitly via pydicom.
    try:
        ds.is_little_endian = True
        ds.is_implicit_VR = False
    except (AttributeError, TypeError):
        pass

    ds.Modality = modality
    ds.PatientID = patient_id
    ds.SeriesInstanceUID = series_uid
    ds.ImageType = image_type or ["ORIGINAL", "PRIMARY", "AXIAL"]
    ds.ImagePositionPatient = [0.0, 0.0, float(z_pos)]
    ds.ImageOrientationPatient = iop or [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelSpacing = pixel_spacing or [0.703125, 0.703125]
    ds.SliceThickness = 2.5
    ds.KVP = 120.0
    ds.Exposure = 150.0
    ds.ConvolutionKernel = "STANDARD"
    ds.RescaleSlope = float(slope)
    ds.RescaleIntercept = float(intercept)
    if pixel_padding_value is not None:
        ds.PixelPaddingValue = int(pixel_padding_value)

    ds.Rows, ds.Columns = int(raw_pixels.shape[0]), int(raw_pixels.shape[1])
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = 16
    ds.BitsStored = int(bits_stored)
    ds.HighBit = int(bits_stored - 1)
    ds.PixelRepresentation = int(pixel_rep)
    ds.PixelData = raw_bytes

    sig = inspect.signature(ds.save_as)
    if "enforce_file_format" in sig.parameters:
        ds.save_as(str(path), enforce_file_format=True)
    else:
        ds.save_as(str(path), write_like_original=False)


def test_compute_slice_axis_coord() -> None:
    class Dummy:
        pass
    d = Dummy()
    d.ImagePositionPatient = [10.0, 20.0, -45.5]
    d.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert math.isclose(compute_slice_axis_coord(d), -45.5, abs_tol=1e-6)
    d.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, -1.0, 0.0]
    assert math.isclose(compute_slice_axis_coord(d), -45.5, abs_tol=1e-6)


def test_to_hu() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "s.dcm"
        raw = np.full((512, 512), 500, dtype=np.int16)
        raw[0, 0] = -2000
        _make_synthetic_dicom(
            p, raw, z_pos=0.0, slope=2.0, intercept=-1024.0, pixel_padding_value=-2000
        )
        ds = pydicom.dcmread(p)
        hu = to_hu(ds)
        assert hu.shape == (512, 512)
        assert hu.dtype == np.float32
        assert np.isclose(hu[0, 0], -1000.0)
        assert np.allclose(hu[1:, 1:], -24.0)

        p_uint = Path(tmp) / "u.dcm"
        raw_u = np.full((64, 64), 1024, dtype=np.uint16)
        # Test a standard uint16 (16-bit stored) wrapped artifact instead of 12-bit
        raw_u[0, 0] = 63536
        _make_synthetic_dicom(
            p_uint, raw_u, z_pos=0.0, slope=1.0, intercept=-1024.0, pixel_rep=0, bits_stored=16
        )
        hu_u = to_hu(pydicom.dcmread(p_uint))
        assert np.isclose(hu_u[0, 0], -1000.0)
        assert np.allclose(hu_u[1:, 1:], 0.0)


def test_load_sorted_ct_slices_monotonic_z() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw = np.zeros((512, 512), dtype=np.int16)
        z_input = [15.0, -30.5, 0.0, -10.2, 5.5]
        for i, z in enumerate(z_input):
            _make_synthetic_dicom(tmp_path / f"slice_{i}.dcm", raw, z_pos=z)
        _make_synthetic_dicom(tmp_path / "dup.dcm", raw, z_pos=0.0001)
        _make_synthetic_dicom(
            tmp_path / "scout.dcm", raw, z_pos=-2.0, image_type=["ORIGINAL", "PRIMARY", "LOCALIZER"]
        )
        _make_synthetic_dicom(tmp_path / "cxr.dcm", raw, z_pos=1.0, modality="DX")
        (tmp_path / "annotation.xml").write_text("<xml>lidc</xml>")

        slices = load_sorted_ct_slices(tmp_path)
        z_sorted = [float(s.ImagePositionPatient[2]) for s in slices]
        assert len(z_sorted) == 5
        assert all(z_sorted[i] < z_sorted[i + 1] for i in range(len(z_sorted) - 1))
        assert np.allclose(z_sorted, sorted(z_input), atol=1e-4)


def test_has_sufficient_non_air() -> None:
    hu = np.full((512, 512), -1000.0, dtype=np.float32)
    assert not has_sufficient_non_air(hu, air_hu_thresh=-900.0, min_fraction=0.05)
    n_exact_5pct = int(0.05 * hu.size)
    hu.ravel()[:n_exact_5pct] = 50.0
    assert has_sufficient_non_air(hu, air_hu_thresh=-900.0, min_fraction=0.05)
    rng = np.random.default_rng(42)
    noisy_air = rng.normal(loc=-1000.0, scale=35.0, size=(512, 512)).astype(np.float32)
    assert not has_sufficient_non_air(noisy_air, air_hu_thresh=-900.0, min_fraction=0.05)


def test_window_and_normalize() -> None:
    hu = np.array([[-1500.0, -1000.0], [-300.0, 800.0]], dtype=np.float32)
    norm = window_and_normalize(hu, hu_min=-1000.0, hu_max=400.0)
    assert np.allclose(norm, [[0.0, 0.0], [0.5, 1.0]])


def test_resize_slice() -> None:
    rng = np.random.default_rng(0)
    norm = rng.uniform(0.0, 1.0, size=(512, 512)).astype(np.float32)
    resized = resize_slice(norm, out_shape=(256, 256))
    assert resized.shape == (256, 256)
    assert 0.0 <= float(resized.min()) and float(resized.max()) <= 1.0


def test_process_patient_series_and_dataset_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        in_dir = Path(tmp) / "in"
        out_dir = Path(tmp) / "out"
        in_dir.mkdir()

        raw_air = np.zeros((512, 512), dtype=np.int16)
        _make_synthetic_dicom(in_dir / "a.dcm", raw_air, z_pos=10.0, slope=1.0, intercept=-1024.0)

        raw_tissue_1 = np.full((512, 512), 1024, dtype=np.int16)
        _make_synthetic_dicom(
            in_dir / "b.dcm", raw_tissue_1, z_pos=-5.0, slope=1.0, intercept=-1024.0, pixel_spacing=[0.7, 0.7]
        )

        raw_tissue_2 = np.full((512, 512), 1424, dtype=np.int16)
        _make_synthetic_dicom(
            in_dir / "c.dcm", raw_tissue_2, z_pos=2.0, slope=1.0, intercept=-1024.0, pixel_spacing=[0.7, 0.7]
        )

        saved = process_patient_series(in_dir, out_dir, patient_id="P01")
        assert len(saved) == 2

        manifest_file = out_dir / "P01" / "manifest.json"
        assert manifest_file.is_file()
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert manifest["num_raw_slices"] == 3
        assert manifest["num_kept_slices"] == 2
        assert np.allclose(manifest["slices"][0]["out_pixel_spacing_mm"], [1.4, 1.4])

        ds_full = LIDCSliceDataset(out_dir, patient_ids=["P01"], return_metadata=True)
        assert len(ds_full) == 2
        t0, meta0 = ds_full[0]
        assert t0.shape == (1, 256, 256)
        assert torch.allclose(t0, torch.full_like(t0, 1000.0 / 1400.0), atol=1e-4)


def test_two_patients_mixed_in_flat_folder() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        flat_in = Path(tmp) / "flat_in"
        out_dir = Path(tmp) / "out"
        flat_in.mkdir()

        raw_p1 = np.full((512, 512), 1024, dtype=np.int16)
        raw_p2 = np.full((512, 512), 1424, dtype=np.int16)

        for i, z in enumerate([10.0, -5.0]):
            _make_synthetic_dicom(
                flat_in / f"p1_{i}.dcm", raw_p1, z_pos=z, slope=1.0, intercept=-1024.0,
                patient_id="LIDC-IDRI-0001", series_uid="SERIES_P1"
            )
        for i, z in enumerate([4.0, -8.0, 0.0]):
            _make_synthetic_dicom(
                flat_in / f"p2_{i}.dcm", raw_p2, z_pos=z, slope=1.0, intercept=-1024.0,
                patient_id="LIDC-IDRI-0002", series_uid="SERIES_P2"
            )

        outputs = process_folder(flat_in, out_dir, max_slices=40)
        assert set(outputs.keys()) == {"LIDC-IDRI-0001", "LIDC-IDRI-0002"}
        assert len(outputs["LIDC-IDRI-0001"]) == 2
        assert len(outputs["LIDC-IDRI-0002"]) == 3
        assert np.allclose(np.load(outputs["LIDC-IDRI-0001"][0]), 1000.0 / 1400.0, atol=1e-4)
        assert np.allclose(np.load(outputs["LIDC-IDRI-0002"][0]), 1.0, atol=1e-4)


def test_series_selection_3_vs_5_picks_5() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        in_dir = Path(tmp) / "in"
        out_dir = Path(tmp) / "out"
        in_dir.mkdir()

        raw_s3 = np.full((512, 512), 724, dtype=np.int16)
        raw_s5 = np.full((512, 512), 1424, dtype=np.int16)

        for i, z in enumerate([0.0, 5.0, 10.0]):
            _make_synthetic_dicom(
                in_dir / f"s3_{i}.dcm", raw_s3, z_pos=z, slope=1.0, intercept=-1024.0,
                patient_id="P_MULTI_SERIES", series_uid="UID_SMALL_3"
            )
        expected_z5 = [-20.0, -10.0, 0.0, 10.0, 20.0]
        for i, z in enumerate([10.0, -20.0, 20.0, 0.0, -10.0]):
            _make_synthetic_dicom(
                in_dir / f"s5_{i}.dcm", raw_s5, z_pos=z, slope=1.0, intercept=-1024.0,
                patient_id="P_MULTI_SERIES", series_uid="UID_LARGE_5"
            )

        slices = load_sorted_ct_slices(in_dir)
        assert len(slices) == 5
        outputs = process_folder(in_dir, out_dir, max_slices=40)
        assert len(outputs["P_MULTI_SERIES"]) == 5
        assert np.allclose(np.load(outputs["P_MULTI_SERIES"][0]), 1.0, atol=1e-4)


def test_process_folder_cap_evenly_spaced_and_z_sorted() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        in_dir = Path(tmp) / "in"
        out_dir = Path(tmp) / "out"
        in_dir.mkdir()

        raw_air = np.zeros((512, 512), dtype=np.int16)
        valid_z = [float(k * 10) for k in range(10)]
        for k, z in enumerate(reversed(valid_z)):
            raw_val = int(1024 - 300 + z)
            raw_tissue = np.full((512, 512), raw_val, dtype=np.int16)
            _make_synthetic_dicom(
                in_dir / f"valid_{k}.dcm", raw_tissue, z_pos=z, slope=1.0, intercept=-1024.0, patient_id="P_CAP"
            )

        _make_synthetic_dicom(
            in_dir / "dup_z0.dcm", np.full((512, 512), 724, dtype=np.int16), z_pos=0.0002, slope=1.0, intercept=-1024.0, patient_id="P_CAP"
        )
        _make_synthetic_dicom(in_dir / "air_neg.dcm", raw_air, z_pos=-50.0, slope=1.0, intercept=-1024.0, patient_id="P_CAP")
        _make_synthetic_dicom(in_dir / "air_pos.dcm", raw_air, z_pos=500.0, slope=1.0, intercept=-1024.0, patient_id="P_CAP")

        outputs = process_folder(in_dir, out_dir, max_slices=4)
        saved_paths = outputs["P_CAP"]
        assert len(saved_paths) == 4

        expected_kept_z = [0.0, 30.0, 60.0, 90.0]
        manifest = json.loads((out_dir / "P_CAP" / "manifest.json").read_text(encoding="utf-8"))
        kept_z = [s["z_mm"] for s in manifest["slices"]]
        assert np.allclose(kept_z, expected_kept_z, atol=1e-3)
        assert all(kept_z[i] < kept_z[i + 1] for i in range(len(kept_z) - 1))

        for path, z in zip(saved_paths, expected_kept_z):
            arr = np.load(path)
            assert np.allclose(arr, (700.0 + z) / 1400.0, atol=1e-4)


if __name__ == "__main__":
    test_compute_slice_axis_coord()
    test_to_hu()
    test_load_sorted_ct_slices_monotonic_z()
    test_has_sufficient_non_air()
    test_window_and_normalize()
    test_resize_slice()
    test_process_patient_series_and_dataset_end_to_end()
    test_two_patients_mixed_in_flat_folder()
    test_series_selection_3_vs_5_picks_5()
    test_process_folder_cap_evenly_spaced_and_z_sorted()
    print("All 10 unit, shape, and adversarial tests in loader.py passed.")