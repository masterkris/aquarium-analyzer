"""
aquarium_augs.py — Albumentations augmentation pipeline for aquarium fish detection.

Targets failure modes specific to phone photos of aquarium tanks:
  - Camera blur (motion, out-of-focus)
  - Sensor noise (ISO grain, Gaussian)
  - JPEG compression artefacts / low-res upscaling
  - Water turbidity / fogging

Intentionally excludes: spatial transforms, flips, mosaic — those are already
handled by YOLO's built-in augmentation and must not be doubled.
"""

import albumentations as A


def build_aquarium_augmentations() -> A.Compose:
    """
    Return an Albumentations pipeline covering aquarium-specific failure modes.

    bbox_params: YOLO format (x_center, y_center, width, height), all normalised.
    Boxes with less than 30% area remaining after a crop/resize are dropped.
    """
    return A.Compose(
        [
            # --- Blur (phone shake, glass refraction, out-of-focus tank shots) ---
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=(3, 15), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 9), p=1.0),
                    A.Defocus(radius=(1, 5), p=1.0),
                ],
                p=0.4,
            ),

            # --- Sensor noise (high-ISO phone shots in dim aquarium lighting) ---
            A.OneOf(
                [
                    A.ISONoise(intensity=(0.1, 0.5), p=1.0),
                    A.GaussNoise(std_range=(0.04, 0.22), p=1.0),
                ],
                p=0.3,
            ),

            # --- Compression / resolution (WhatsApp-forwarded tank photos, screenshots) ---
            A.ImageCompression(quality_range=(40, 90), p=0.3),
            A.Downscale(scale_range=(0.5, 0.9), p=0.15),

            # --- Color / lighting (dim tanks, LED glare, freshwater vs saltwater tint) ---
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=15, p=0.3),
            A.CLAHE(clip_limit=2, p=0.15),
            A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.1),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )
