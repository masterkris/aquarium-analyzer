"""
generate_augmented_dataset.py — Pre-generate an augmented YOLO dataset.

For each train image: writes the original as {stem}.jpg/.txt, then writes
{stem}_aug{i}.jpg/.txt for i in range(n_augs).  valid/ and test/ are copied
through unchanged.  Augmentations that drop every bounding box are skipped.

Usage:
    py -3.12 generate_augmented_dataset.py --src <dir> --dst <dir> --n-augs 3
        [--dry-run] [--limit N] [--seed 42] [--overwrite]

Dry-run: no dataset is written; instead, side-by-side preview PNGs
(original | aug0 | aug1 | ...) with bboxes drawn are saved to
_debug_previews/ at the repo root for the first 20 train images.
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from aquarium_augs import build_aquarium_augmentations

_REPO_ROOT = Path(__file__).resolve().parents[4]   # …/Aquarium_Analyser
_DEBUG_DIR = _REPO_ROOT / "_debug_previews"
_MAX_PREVIEW = 20
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 95]


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _parse_label(label_path: Path) -> tuple[list[int], list[tuple[float, ...]]]:
    """Return (class_ids, bboxes) from a YOLO label file."""
    class_ids: list[int] = []
    bboxes: list[tuple[float, ...]] = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                class_ids.append(int(parts[0]))
                bboxes.append(tuple(float(x) for x in parts[1:]))
    return class_ids, bboxes


def _write_label(
    label_path: Path,
    class_ids: list[int],
    bboxes: list[tuple[float, ...]],
) -> None:
    lines = [
        f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        for c, (cx, cy, w, h) in zip(class_ids, bboxes)
    ]
    label_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Preview helpers
# ---------------------------------------------------------------------------

def _draw_bboxes(
    image_rgb: np.ndarray,
    bboxes: list[tuple[float, ...]],
) -> np.ndarray:
    img = image_rgb.copy()
    ih, iw = img.shape[:2]
    for cx, cy, bw, bh in bboxes:
        x1 = int((cx - bw / 2) * iw)
        y1 = int((cy - bh / 2) * ih)
        x2 = int((cx + bw / 2) * iw)
        y2 = int((cy + bh / 2) * ih)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img


def _side_by_side(panels: list[np.ndarray]) -> np.ndarray:
    target_h = max(p.shape[0] for p in panels)
    out = []
    for p in panels:
        if p.shape[0] != target_h:
            scale = target_h / p.shape[0]
            p = cv2.resize(p, (int(p.shape[1] * scale), target_h))
        out.append(p)
    return np.concatenate(out, axis=1)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def _process_train(
    src_img_dir: Path,
    src_lbl_dir: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    n_augs: int,
    base_seed: int,
    dry_run: bool,
    limit: int | None,
) -> tuple[int, int, int]:
    """Return (n_originals, n_augs_written, n_augs_skipped)."""
    if not src_img_dir.exists():
        sys.exit(f"Error: train/images dir not found: {src_img_dir}")

    compose = build_aquarium_augmentations()

    img_paths = sorted(
        p for p in src_img_dir.iterdir() if p.suffix.lower() in _IMG_EXTS
    )
    if limit is not None:
        img_paths = img_paths[:limit]

    if not dry_run:
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    else:
        _DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    n_orig = 0
    n_written = 0
    n_skipped = 0

    for img_idx, img_path in enumerate(tqdm(img_paths, desc="train", unit="img")):
        stem = img_path.stem
        lbl_path = src_lbl_dir / (stem + ".txt")

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            tqdm.write(f"  [warn] cannot read {img_path.name}, skipping")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        class_ids, bboxes = _parse_label(lbl_path)

        # --- original ---
        if not dry_run:
            cv2.imwrite(str(dst_img_dir / f"{stem}.jpg"), bgr, _JPEG_PARAMS)
            _write_label(dst_lbl_dir / f"{stem}.txt", class_ids, bboxes)
        n_orig += 1

        preview_panels: list[np.ndarray] = []
        make_preview = dry_run and img_idx < _MAX_PREVIEW
        if make_preview:
            preview_panels.append(_draw_bboxes(rgb, bboxes))

        # --- augmented copies ---
        for aug_i in range(n_augs):
            rng_seed = (base_seed + img_idx * 10_000 + aug_i) & 0xFFFF_FFFF
            random.seed(rng_seed)
            np.random.seed(rng_seed)

            result = compose(
                image=rgb,
                bboxes=list(bboxes),
                class_labels=list(class_ids),
            )
            aug_bboxes: list[tuple[float, ...]] = result["bboxes"]
            aug_labels: list[int] = result["class_labels"]

            if not aug_bboxes:
                n_skipped += 1
                continue  # all boxes dropped — skip this aug

            aug_rgb: np.ndarray = result["image"]

            if not dry_run:
                aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(dst_img_dir / f"{stem}_aug{aug_i}.jpg"),
                    aug_bgr,
                    _JPEG_PARAMS,
                )
                _write_label(
                    dst_lbl_dir / f"{stem}_aug{aug_i}.txt",
                    aug_labels,
                    aug_bboxes,
                )
            n_written += 1

            if make_preview:
                preview_panels.append(_draw_bboxes(aug_rgb, aug_bboxes))

        if make_preview and preview_panels:
            preview_bgr = cv2.cvtColor(_side_by_side(preview_panels), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(_DEBUG_DIR / f"{stem}_preview.png"), preview_bgr)

    return n_orig, n_written, n_skipped


def _copy_split(src_split: Path, dst_split: Path) -> None:
    if not src_split.exists():
        return
    shutil.copytree(src_split, dst_split)
    print(f"  Copied {src_split.name}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pre-generate an augmented YOLO dataset using aquarium_augs.py."
    )
    ap.add_argument("--src", required=True, type=Path, help="Source YOLO dataset root")
    ap.add_argument("--dst", required=True, type=Path, help="Destination root")
    ap.add_argument("--n-augs", type=int, default=3, help="Augmented copies per train image (default 3)")
    ap.add_argument("--dry-run", action="store_true", help="Skip dataset write; save preview PNGs instead")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N train images")
    ap.add_argument("--seed", type=int, default=42, help="Base RNG seed (default 42)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite non-empty --dst")
    args = ap.parse_args()

    src: Path = args.src.resolve()
    dst: Path = args.dst.resolve()

    if not src.exists():
        sys.exit(f"Error: --src {src} does not exist")

    src_yaml = src / "data.yaml"
    if not src_yaml.exists():
        sys.exit(f"Error: no data.yaml found at {src_yaml}")

    if not args.dry_run:
        if dst.exists() and any(dst.iterdir()):
            if not args.overwrite:
                sys.exit(
                    f"Error: --dst {dst} is non-empty. Pass --overwrite to proceed."
                )
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)

    src_meta: dict = yaml.safe_load(src_yaml.read_text())
    nc = src_meta.get("nc")
    names = src_meta.get("names")

    print(f"Source : {src}")
    print(f"Dest   : {dst}  (dry-run={args.dry_run})")
    print(f"n-augs : {args.n_augs}  seed={args.seed}  limit={args.limit}")

    n_orig, n_written, n_skipped = _process_train(
        src_img_dir=src / "train" / "images",
        src_lbl_dir=src / "train" / "labels",
        dst_img_dir=dst / "train" / "images",
        dst_lbl_dir=dst / "train" / "labels",
        n_augs=args.n_augs,
        base_seed=args.seed,
        dry_run=args.dry_run,
        limit=args.limit,
    )

    total_train = n_orig + n_written
    print(
        f"\nTrain: {n_orig} originals + {n_written} augs = {total_train} total"
        + (f"  ({n_skipped} augs skipped — all boxes dropped)" if n_skipped else "")
    )

    if not args.dry_run:
        for split in ("valid", "test"):
            _copy_split(src / split, dst / split)

        dst_meta = {
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": nc,
            "names": names,
        }
        with open(dst / "data.yaml", "w") as fh:
            yaml.dump(dst_meta, fh, default_flow_style=False, sort_keys=False)
        print(f"Written data.yaml → {dst / 'data.yaml'}")
    else:
        print(f"Dry-run complete. Previews saved to {_DEBUG_DIR}")


if __name__ == "__main__":
    main()
