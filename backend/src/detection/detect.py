"""
detect.py — Fish Detection Module
Aquarium Analyzer | Midpoint Demo (March 25)

Training strategy: combined dataset
  1. Kaggle (zehraatlgan/fish-detection)
       8,242 images | 13 species-level classes | YOLO format
       Strong on: species shape diversity, clean per-species bounding boxes
       Weak on:   real aquarium tank backgrounds

  2. Roboflow Aquarium Combined (brad-dwyer/aquarium-combined)
       638 images | 7 mixed classes | YOLO format
       Strong on: cluttered backgrounds, partial occlusion, uneven lighting
       Weak on:   small size, no species-level labels (just "fish")

  Combined: ~8,880 images. Kaggle teaches species shapes, Roboflow teaches
  the model to find fish in real messy aquarium conditions.

Inference:
  - Model detects all fish classes from the merged label set
  - Only returns detections whose class maps to an actual fish
    (drops jellyfish, penguin, puffin, starfish, stingray from Roboflow)
  - Passes cropped fish images to classify.py for species identification

Target species for midpoint demo:
  Goldfish | Clownfish | Platy | Angelfish | Yellow Tang
"""

import os
import shutil
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Roboflow Aquarium Combined — original 7 classes (index order matters for labels)
ROBOFLOW_CLASSES = [
    "fish",        # index 0
    "jellyfish",   # index 1 — not a fish, filtered at inference
    "penguin",     # index 2 — not a fish, filtered at inference
    "puffin",      # index 3 — not a fish, filtered at inference
    "shark",       # index 4 — not a target species, filtered at inference
    "starfish",    # index 5 — not a fish, filtered at inference
    "stingray",    # index 6 — not a fish, filtered at inference
]

# Kaggle fish-detection — 13 species-level classes (index order matters for labels)
KAGGLE_CLASSES = [
    "AngelFish",              # index 0
    "BlueTang",               # index 1
    "ButterflyFish",          # index 2
    "ClownFish",              # index 3
    "GoldFish",               # index 4
    "Gourami",                # index 5
    "MorishIdol",             # index 6
    "PlatyFish",              # index 7
    "RibbonedSweetlips",      # index 8
    "ThreeStripedDamselfish", # index 9
    "YellowCichlid",          # index 10
    "YellowTang",             # index 11
    "ZebraFish",              # index 12
]

# Unified class list for the merged dataset.
# Roboflow's indices 0-6 stay unchanged.
# Kaggle's indices are shifted up by 7 (class 0 → 7, class 1 → 8, etc.)
MERGED_CLASSES = [
    # --- from Roboflow (indices 0-6) ---
    "fish",                   # 0  generic / unidentified fish
    "jellyfish",              # 1  filtered at inference
    "penguin",                # 2  filtered at inference
    "puffin",                 # 3  filtered at inference
    "shark",                  # 4  filtered at inference
    "starfish",               # 5  filtered at inference
    "stingray",               # 6  filtered at inference
    # --- from Kaggle (indices 7-19) ---
    "AngelFish",              # 7
    "BlueTang",               # 8
    "ButterflyFish",          # 9
    "ClownFish",              # 10
    "GoldFish",               # 11
    "Gourami",                # 12
    "MorishIdol",             # 13
    "PlatyFish",              # 14
    "RibbonedSweetlips",      # 15
    "ThreeStripedDamselfish", # 16
    "YellowCichlid",          # 17
    "YellowTang",             # 18
    "ZebraFish",              # 19
]

# Classes we actually RETURN from detect_fish().
# jellyfish, penguin, puffin, shark, starfish, stingray are dropped silently.
FISH_CLASSES = {
    "fish",
    "AngelFish", "BlueTang", "ButterflyFish", "ClownFish", "GoldFish",
    "Gourami", "MorishIdol", "PlatyFish", "RibbonedSweetlips",
    "ThreeStripedDamselfish", "YellowCichlid", "YellowTang", "ZebraFish",
}

# Midpoint demo target species (March 25).
# detect.py crops fish → classify.py identifies down to this level.
TARGET_SPECIES = ["GoldFish", "ClownFish", "PlatyFish", "AngelFish", "YellowTang"]

# Default inference thresholds
DEFAULT_CONF = 0.35
DEFAULT_IOU  = 0.45
MIN_BOX_SIZE = 20   # pixels — filters out noise detections


# ---------------------------------------------------------------------------
# Dataset merge utility
# ---------------------------------------------------------------------------

def merge_datasets(
    roboflow_dir: str,
    kaggle_dir: str,
    output_dir: str = "data/merged",
) -> str:
    """
    Merge the Roboflow Aquarium Combined and Kaggle fish-detection datasets
    into a single unified YOLO dataset ready for training.

    What this does
    --------------
    Both datasets use YOLO format (.txt label files) but have different
    class index mappings. We need to:
      1. Copy all images and label files into one folder structure
      2. Re-index Kaggle label files (class 0 → 7, class 1 → 8, ... class 12 → 19)
         so they align with MERGED_CLASSES
      3. Leave Roboflow label files unchanged (indices 0-6 already correct)
      4. Write a unified data.yaml that YOLO can train from directly

    Parameters
    ----------
    roboflow_dir : str
        Root of the downloaded Roboflow dataset.
        Expected structure:
            roboflow_dir/
              train/images/  train/labels/
              valid/images/  valid/labels/
              test/images/   test/labels/
              data.yaml

    kaggle_dir : str
        Root of the downloaded Kaggle dataset (same structure as above).

    output_dir : str
        Where to write the merged dataset (created if needed).

    Returns
    -------
    str
        Path to the merged data.yaml — pass directly to train_combined().

    Usage
    -----
        yaml_path = merge_datasets(
            roboflow_dir="datasets/aquarium-combined",
            kaggle_dir="datasets/fish-detection",
        )
        train_combined(yaml_path)
    """
    output_dir = Path(output_dir)
    splits = ["train", "valid", "test"]

    print(f"[detect.py] Merging datasets into: {output_dir}")

    for split in splits:
        for sub in ["images", "labels"]:
            (output_dir / split / sub).mkdir(parents=True, exist_ok=True)

    # Roboflow — no re-indexing needed (indices 0-6 stay the same)
    _copy_dataset(Path(roboflow_dir), output_dir, splits, label_offset=0, tag="rf")

    # Kaggle — shift all class ids up by 7
    _copy_dataset(Path(kaggle_dir), output_dir, splits, label_offset=7, tag="kg")

    # Write unified data.yaml
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(
            {
                "path":  str(output_dir.resolve()),
                "train": "train/images",
                "val":   "valid/images",
                "test":  "test/images",
                "nc":    len(MERGED_CLASSES),
                "names": MERGED_CLASSES,
            },
            f, default_flow_style=False, sort_keys=False,
        )

    for split in splits:
        n = len(list((output_dir / split / "images").glob("*")))
        print(f"  {split:6s} → {n:5d} images")

    print(f"[detect.py] Merged data.yaml → {yaml_path}")
    return str(yaml_path)


def _copy_dataset(src_dir: Path, dst_dir: Path, splits: list, label_offset: int, tag: str):
    """Copy images + re-indexed labels from src into dst. Prefix filenames with tag."""
    for split in splits:
        src_imgs = src_dir / split / "images"
        src_lbls = src_dir / split / "labels"
        dst_imgs = dst_dir / split / "images"
        dst_lbls = dst_dir / split / "labels"

        if not src_imgs.exists():
            print(f"  [warn] {src_imgs} not found — skipping")
            continue

        for img in src_imgs.iterdir():
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            new_name = f"{tag}_{img.name}"
            shutil.copy2(img, dst_imgs / new_name)

            lbl = src_lbls / (img.stem + ".txt")
            dst_lbl = dst_lbls / (Path(new_name).stem + ".txt")

            if lbl.exists():
                _reindex_label_file(lbl, dst_lbl, label_offset)
            else:
                dst_lbl.touch()


def _reindex_label_file(src: Path, dst: Path, offset: int):
    """
    Rewrite a YOLO label file with class ids shifted by offset.
    YOLO format per line: <class_id> <x_center> <y_center> <width> <height>
    """
    lines_out = []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts  = line.split()
            cls_id = int(parts[0]) + offset
            lines_out.append(f"{cls_id} {' '.join(parts[1:])}")
    with open(dst, "w") as f:
        f.write("\n".join(lines_out))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_combined(
    merged_yaml: str,
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    img_size: int = 640,
    output_name: str = "aquarium_combined_yolov8",
):
    """
    Fine-tune YOLOv8 on the merged Kaggle + Roboflow dataset.

    Parameters
    ----------
    merged_yaml : str
        Path to data.yaml produced by merge_datasets().
    base_model : str
        Starting weights. 'yolov8n.pt' = COCO pretrained nano model.
        Use yolov8s.pt / yolov8m.pt for better accuracy at cost of speed.
    epochs : int
        50 is a reasonable starting point for ~8,880 images.
    output_name : str
        Weights saved to runs/detect/<output_name>/weights/best.pt

    Augmentation notes
    ------------------
    Kaggle already applied: rotation, exposure jitter, Gaussian blur (2 versions).
    We add augmentations for what Kaggle did NOT cover:
      - HSV jitter    → aquarium colour temperature shifts
      - Mosaic        → compensates for Roboflow's small 638-image portion
      - Copy-paste    → simulates multiple fish overlapping in a tank
      - Scale/shear   → partial occlusion by plants, rocks, tank edges

    Usage
    -----
        train_combined("data/merged/data.yaml")
        # or via CLI:
        # python detect.py train data/merged/data.yaml
    """
    model = YOLO(base_model)
    model.train(
        data=merged_yaml,
        epochs=epochs,
        imgsz=img_size,
        name=output_name,
        hsv_h=0.015,       # hue jitter — aquarium lighting colour shift
        hsv_s=0.7,         # saturation jitter
        hsv_v=0.4,         # brightness jitter — uneven tank lighting
        fliplr=0.5,        # horizontal flip — fish swim both ways
        flipud=0.0,        # fish are almost never upside-down
        degrees=15.0,      # matches Kaggle's ±15° rotation augmentation
        translate=0.1,
        scale=0.5,         # scale jitter — fish at different depths/distances
        shear=2.0,
        perspective=0.0005,
        mosaic=1.0,        # paste 4 images together — critical for small Roboflow portion
        mixup=0.1,
        copy_paste=0.1,    # copy fish across images — simulates crowded tanks
    )
    print(
        f"\n[detect.py] Training complete!\n"
        f"  Best weights → runs/detect/{output_name}/weights/best.pt\n"
        f"  Use: python detect.py detect <image> --model runs/detect/{output_name}/weights/best.pt"
    )


# ---------------------------------------------------------------------------
# Core detection function
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS = str(Path(__file__).parent / "best.pt")


def detect_fish(
    image_path: str,
    model_path: str = _DEFAULT_WEIGHTS,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    use_tta: bool = False,
    save_crops: bool = False,
    crops_dir: str = "crops",
) -> list[dict]:
    """
    Run YOLOv8 fish detection on a single aquarium image.

    Returns
    -------
    list[dict]  — one entry per detected fish:
        {
            "box":        [x1, y1, x2, y2],  # pixel coords (ints)
            "conf":       float,              # detection confidence 0-1
            "class_id":   int,               # index in MERGED_CLASSES
            "class_name": str,               # e.g. "ClownFish", "GoldFish", "fish"
            "crop":       np.ndarray,         # BGR image crop → pass to classify.py
            "crop_path":  str | None,         # file path if save_crops=True
        }

    Non-fish detections (jellyfish, penguin, puffin, shark, starfish, stingray)
    are silently dropped — only actual fish are returned.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"cv2 could not read image: {image_path}")

    model = YOLO(model_path)

    # Inference — augment=True enables ultralytics' built-in TTA (flip + scale)
    result = model.predict(
        source=str(image_path), conf=conf, iou=iou,
        augment=use_tta, verbose=False
    )[0]

    detections = []

    if result.boxes is None or len(result.boxes) == 0:
        print(f"[detect.py] No fish detected in: {image_path}")
        return detections

    boxes_data = result.boxes.xyxy.cpu().numpy()
    confs_data = result.boxes.conf.cpu().numpy()
    cls_data   = result.boxes.cls.cpu().numpy().astype(int)

    if save_crops:
        Path(crops_dir).mkdir(parents=True, exist_ok=True)

    img_h, img_w = image.shape[:2]
    stem = Path(image_path).stem

    for i, (box, conf_val, cls_id) in enumerate(zip(boxes_data, confs_data, cls_data)):
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        if (x2 - x1) < MIN_BOX_SIZE or (y2 - y1) < MIN_BOX_SIZE:
            continue

        class_name = (
            MERGED_CLASSES[cls_id]
            if cls_id < len(MERGED_CLASSES)
            else result.names.get(cls_id, f"class_{cls_id}")
        )

        # Drop non-fish classes
        if class_name not in FISH_CLASSES:
            continue

        crop = image[y1:y2, x1:x2]

        crop_path = None
        if save_crops:
            crop_path = os.path.join(crops_dir, f"{stem}_det{i}_{class_name}.jpg")
            cv2.imwrite(crop_path, crop)

        detections.append({
            "box":        [x1, y1, x2, y2],
            "conf":       float(conf_val),
            "class_id":   int(cls_id),
            "class_name": class_name,
            "crop":       crop,
            "crop_path":  crop_path,
        })

    return detections


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw_detections(
    image_path: str,
    detections: list[dict],
    output_path: str = "output_detected.jpg",
) -> str:
    """Draw bounding boxes on the image and save. Returns the output path."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f"{det['class_name']} {det['conf']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 50), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 200, 50), -1)
        cv2.putText(image, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, image)
    print(f"[detect.py] Annotated image saved → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    parser = argparse.ArgumentParser(
        description="Aquarium fish detector — YOLOv8 on combined Kaggle + Roboflow dataset."
    )
    sub = parser.add_subparsers(dest="command")

    # detect
    d = sub.add_parser("detect", help="Run detection on an image")
    d.add_argument("image")
    d.add_argument("--model",      default=_DEFAULT_WEIGHTS)
    d.add_argument("--conf",       type=float, default=DEFAULT_CONF)
    d.add_argument("--iou",        type=float, default=DEFAULT_IOU)
    d.add_argument("--tta",        action="store_true")
    d.add_argument("--save-crops", action="store_true")
    d.add_argument("--crops-dir",  default="crops")
    d.add_argument("--draw",       action="store_true")
    d.add_argument("--output",     default="output_detected.jpg")

    # merge
    m = sub.add_parser("merge", help="Merge Kaggle + Roboflow into one dataset")
    m.add_argument("--roboflow", required=True, help="Path to Roboflow dataset root")
    m.add_argument("--kaggle",   required=True, help="Path to Kaggle dataset root")
    m.add_argument("--output",   default="data/merged")

    # train
    t = sub.add_parser("train", help="Train YOLOv8 on merged dataset")
    t.add_argument("merged_yaml",  help="Path to merged data.yaml")
    t.add_argument("--base-model", default="yolov8n.pt")
    t.add_argument("--epochs",     type=int, default=50)
    t.add_argument("--img-size",   type=int, default=640)
    t.add_argument("--name",       default="aquarium_combined_yolov8")

    return parser


def main():
    args = _build_parser().parse_args()

    if args.command == "merge":
        yaml_path = merge_datasets(args.roboflow, args.kaggle, args.output)
        print(f"\nNext: python detect.py train {yaml_path}")

    elif args.command == "train":
        train_combined(args.merged_yaml, args.base_model, args.epochs, args.img_size, args.name)

    else:  # detect (default)
        if not hasattr(args, "image"):
            _build_parser().print_help()
            return

        print(f"[detect.py] Image : {args.image}")
        print(f"[detect.py] Model : {args.model}  conf={args.conf}  iou={args.iou}  tta={args.tta}")

        detections = detect_fish(
            image_path=args.image,
            model_path=args.model,
            conf=args.conf,
            iou=args.iou,
            use_tta=args.tta,
            save_crops=args.save_crops,
            crops_dir=args.crops_dir,
        )

        print(f"\n[detect.py] Found {len(detections)} fish:\n")
        for i, det in enumerate(detections):
            print(
                f"  [{i}] {det['class_name']:25s}  conf={det['conf']:.3f}"
                f"  box={det['box']}"
                + (f"  crop → {det['crop_path']}" if det["crop_path"] else "")
            )

        if args.draw and detections:
            draw_detections(args.image, detections, args.output)

        if not detections:
            print("  (no detections — try lowering --conf or enabling --tta)")


if __name__ == "__main__":
    main()