#!/usr/bin/env python3
"""
mapper.py — Phase 1: MediaPipe Mapper (Masking)
================================================
Extracts high-quality semantic masks using MediaPipe:
  1. Face/Skin mask via SelfieSegmentation + FaceLandmarker
  2. Background mask (inverse of person segmentation)

Saves mask images to test_data/masks/.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(name)s]  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("APS.Mapper")

# ── MediaPipe model URLs ──────────────────────────────────────────────────
_MP_MODEL_BASE = "https://storage.googleapis.com/mediapipe-models"
_SEGMENTER_URL = (
    f"{_MP_MODEL_BASE}/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite"
)
_SEGMENTER_LANDSCAPE_URL = (
    f"{_MP_MODEL_BASE}/image_segmenter/selfie_segmenter_landscape/float16/1/"
    "selfie_segmenter_landscape.tflite"
)
_LANDMARKER_URL = (
    f"{_MP_MODEL_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# ── FaceMesh landmark groups ─────────────────────────────────────────────
_FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_LIPS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78,
]
_LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
_RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]


def _ensure_model(url: str, cache_dir: Path) -> str:
    """Download a MediaPipe model file if not already cached. Returns local path."""
    filename = url.rsplit("/", 1)[-1]
    local_path = cache_dir / filename
    if local_path.exists():
        log.info("Model cached: %s", local_path)
        return str(local_path)
    log.info("Downloading model %s …", filename)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    local_path.write_bytes(resp.content)
    log.info("Saved model → %s", local_path)
    return str(local_path)


def _landmarks_to_mask(
    h: int, w: int, landmarks, indices: list
) -> np.ndarray:
    """Draw a filled polygon from FaceLandmarker landmark indices → binary mask."""
    pts = []
    for idx in indices:
        lm = landmarks[idx]
        pts.append((int(lm.x * w), int(lm.y * h)))
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    return mask


class Mapper:
    """
    MediaPipe-based semantic mapper.
    Produces:
      - skin_mask: face skin regions (face oval minus eyes/brows/lips)
      - bg_mask: background (inverse of person segmentation)
    """

    def __init__(self, model_cache_dir: Optional[str] = None):
        import mediapipe as mp

        cache_dir = Path(model_cache_dir or ".aps_cache/models")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # ── Selfie Segmenter (person vs background) ──
        seg_model_path = _ensure_model(_SEGMENTER_LANDSCAPE_URL, cache_dir)
        seg_base = mp.tasks.BaseOptions(model_asset_path=seg_model_path)
        seg_options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=seg_base,
            output_category_mask=True,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        self.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(seg_options)
        log.info("ImageSegmenter ready")

        # ── Face Landmarker (468/478 landmarks) ──
        lm_model_path = _ensure_model(_LANDMARKER_URL, cache_dir)
        lm_base = mp.tasks.BaseOptions(model_asset_path=lm_model_path)
        lm_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=lm_base,
            num_faces=10,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(lm_options)
        log.info("FaceLandmarker ready")

    def _person_mask(self, bgr: np.ndarray) -> np.ndarray:
        """Run selfie segmenter → binary person mask (uint8 0/255)."""
        import mediapipe as mp

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.segmenter.segment(mp_image)
        cat_mask = result.category_mask.numpy_view()
        return (cat_mask > 0).astype(np.uint8) * 255

    def _build_skin_mask(self, bgr: np.ndarray) -> np.ndarray:
        """
        Build skin mask using FaceLandmarker.
        Skin = face oval minus eyes, eyebrows, lips.
        """
        import mediapipe as mp

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.face_landmarker.detect(mp_image)

        skin_mask = np.zeros((h, w), dtype=np.uint8)

        if results.face_landmarks:
            for lm in results.face_landmarks:
                # Face oval
                oval = _landmarks_to_mask(h, w, lm, _FACE_OVAL)

                # Exclusion zones: eyes, eyebrows, lips
                exclude = np.zeros((h, w), dtype=np.uint8)
                for region in [_LEFT_EYE, _RIGHT_EYE, _LEFT_EYEBROW,
                               _RIGHT_EYEBROW, _LIPS]:
                    region_mask = _landmarks_to_mask(h, w, lm, region)
                    exclude = cv2.bitwise_or(exclude, region_mask)

                face_skin = cv2.bitwise_and(oval, cv2.bitwise_not(exclude))
                skin_mask = cv2.bitwise_or(skin_mask, face_skin)

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return skin_mask

    def extract_masks(
        self, image_path: str, output_dir: str = "test_data/masks"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full mapper pipeline for a single image.

        Returns:
            (skin_mask, bg_mask) — both uint8, 0/255 binary masks.

        Side effect: saves mask PNGs to output_dir.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        stem = Path(image_path).stem
        log.info("Mapping: %s (%dx%d)", stem, bgr.shape[1], bgr.shape[0])

        # Person/background segmentation
        person_mask = self._person_mask(bgr)
        bg_mask = cv2.bitwise_not(person_mask)

        # Skin mask via landmarks
        skin_mask = self._build_skin_mask(bgr)

        # Combine: skin only where person is detected
        skin_mask = cv2.bitwise_and(skin_mask, person_mask)

        # Save
        skin_path = out / f"{stem}_skin_mask.png"
        bg_path = out / f"{stem}_bg_mask.png"
        cv2.imwrite(str(skin_path), skin_mask)
        cv2.imwrite(str(bg_path), bg_mask)

        skin_pct = float(np.sum(skin_mask > 0)) / skin_mask.size * 100
        bg_pct = float(np.sum(bg_mask > 0)) / bg_mask.size * 100
        log.info("Mapper: %s — skin=%.1f%%, background=%.1f%%", stem, skin_pct, bg_pct)
        log.info("Saved: %s, %s", skin_path, bg_path)

        return skin_mask, bg_mask


if __name__ == "__main__":
    import sys

    input_dir = sys.argv[1] if len(sys.argv) > 1 else "test_data/raws"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "test_data/masks"

    mapper = Mapper()
    for img_file in sorted(Path(input_dir).glob("*")):
        if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
            mapper.extract_masks(str(img_file), output_dir)
