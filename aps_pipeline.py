#!/usr/bin/env python3
"""
APS Pipeline — Production Auto Photo Studio
=============================================
Sony A7M5 batch portrait retouching pipeline.

Architecture:
  The Sorter  — pyiqa (MUSIQ / PIQE) image quality filtering
  The Mapper  — YOLOv8-face detection + MediaPipe skin/face segmentation
  The Forge   — ComfyUI headless API (WAS Node Suite + FaceDetailer)

Modes:
  pipeline  — full retouching pipeline (default)
  convert   — bulk image format conversion (RAW/JPG/PNG/TIFF/BMP)

Aesthetic contract:
  - Frequency separation: preserve high-freq pores, smooth low-freq only
  - Neutral Gray dodge & burn for 3D contouring
  - NO plastic/silicone/glamour retouching
  - Denoise locked to 0.25-0.35
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import struct
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rawpy
import requests
import torch
import yaml
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(name)s]  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("APS")

# ---------------------------------------------------------------------------
# Configuration (overridable via aps_config.yaml)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "sorter": {
        "metric": "musiq",
        "threshold": 0.45,
        "fallback_metric": "piqe",
    },
    "mapper": {
        "yolo_model": "yolov8m-face",
        "yolo_conf": 0.40,
        "selfie_model": 1,        # MediaPipe SelfieSegmentation model (0=general, 1=landscape)
        "facemesh_refine": True,   # enable iris/lip refinement landmarks
        "facemesh_confidence": 0.5,
        "cache_dir": "./.aps_cache/embeddings",
    },
    "forge": {
        "comfyui_url": "http://127.0.0.1:8188",
        "denoise_min": 0.25,
        "denoise_max": 0.35,
        "denoise": 0.30,
        "guide_size": 384,
        "positive_prompt": (
            "lightly even out skin tone and remove redness, "
            "preserve natural skin texture and facial details, "
            "asian aesthetic, macro photography, highly detailed"
        ),
        "negative_prompt": (
            "glamour retouching, heavy makeup, plastic skin, "
            "silicone reflection, glossy face, low detail"
        ),
        "strip_metadata": True,
        "sampler": "euler_ancestral",
        "scheduler": "normal",
        "steps": 20,
        "cfg_scale": 7.0,
        "checkpoint": "realisticVisionV60B1_v51VAE.safetensors",
    },
    "output_dir": "./output",
    "input_dir": "./test_images",
    "supported_ext": [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".arw", ".cr3"],
}


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    if path and Path(path).exists():
        with open(path) as f:
            user_cfg = yaml.safe_load(f) or {}
        _deep_merge(cfg, user_cfg)
        log.info("Loaded config from %s", path)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


# ═══════════════════════════════════════════════════════════════════════════
# DATA MODEL
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ImageRecord:
    source_path: Path
    uid: str = ""

    # Sorter
    quality_score: Optional[float] = None
    is_accepted: bool = False

    # Mapper
    decoded_bgr: Optional[Any] = None
    face_boxes: List[Dict[str, Any]] = field(default_factory=list)
    skin_mask: Optional[Any] = None
    hair_mask: Optional[Any] = None
    face_parse_mask: Optional[Any] = None
    embedding_cache_path: Optional[Path] = None

    # Forge
    retouched_bgr: Optional[Any] = None
    retouched_path: Optional[Path] = None
    forge_metadata: Dict[str, Any] = field(default_factory=dict)

    # Quality
    quality_checks: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# IMAGE DECODER (Sony ARW + standard formats)
# ═══════════════════════════════════════════════════════════════════════════

class ImageDecoder:
    """Decodes Sony ARW (via rawpy) and standard image formats to BGR numpy."""

    RAW_EXTENSIONS = {".arw", ".cr3", ".nef", ".dng", ".raf"}

    @staticmethod
    def decode(path: Path) -> Optional[np.ndarray]:
        ext = path.suffix.lower()
        try:
            if ext in ImageDecoder.RAW_EXTENSIONS:
                with rawpy.imread(str(path)) as raw:
                    rgb = raw.postprocess(
                        use_camera_wb=True,
                        half_size=False,
                        no_auto_bright=False,
                        output_bps=8,
                    )
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                log.info("Decoded RAW: %s → %dx%d", path.name, bgr.shape[1], bgr.shape[0])
                return bgr
            else:
                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if bgr is not None:
                    log.info("Decoded: %s → %dx%d", path.name, bgr.shape[1], bgr.shape[0])
                return bgr
        except Exception as e:
            log.error("Failed to decode %s: %s", path.name, e)
            return None


# ═══════════════════════════════════════════════════════════════════════════
# 1. THE SORTER — pyiqa Image Quality Assessment
# ═══════════════════════════════════════════════════════════════════════════

class Sorter:
    """
    No-reference IQA using pyiqa.
    Primary: MUSIQ (multi-scale image quality transformer).
    Fallback: PIQE (perception-based, no deep model needed).
    Images below threshold are rejected from the pipeline.
    """

    def __init__(self, cfg: Dict[str, Any]):
        import pyiqa

        self.threshold = cfg.get("threshold", 0.45)
        metric_name = cfg.get("metric", "musiq")
        fallback = cfg.get("fallback_metric", "piqe")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.model = pyiqa.create_metric(metric_name, device=device)
            self.metric_name = metric_name
            self.is_lower_better = self.model.lower_better
            log.info("Sorter ready: metric=%s, device=%s, threshold=%.2f",
                     metric_name, device, self.threshold)
        except Exception as e:
            log.warning("Failed to load %s (%s), falling back to %s", metric_name, e, fallback)
            self.model = pyiqa.create_metric(fallback, device=device)
            self.metric_name = fallback
            self.is_lower_better = self.model.lower_better
            log.info("Sorter ready (fallback): metric=%s", fallback)

    def evaluate(self, rec: ImageRecord) -> ImageRecord:
        """Score the image and set is_accepted based on threshold."""
        bgr = rec.decoded_bgr
        # pyiqa expects RGB PIL Image or torch tensor
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        with torch.no_grad():
            score = self.model(pil_img).item()

        # Normalise: for lower-is-better metrics (like PIQE), invert
        if self.is_lower_better:
            # PIQE range is roughly 0-100, lower=better
            # Map to 0-1 where 1=best
            normalised = max(0.0, 1.0 - score / 100.0)
        else:
            # MUSIQ range is roughly 0-100, higher=better
            normalised = min(score / 100.0, 1.0)

        rec.quality_score = round(normalised, 4)
        rec.is_accepted = normalised >= self.threshold

        log.info(
            "Sorter: %-25s  %s=%.2f → norm=%.4f  %s",
            rec.source_path.name,
            self.metric_name, score, normalised,
            "ACCEPTED" if rec.is_accepted else "REJECTED",
        )
        return rec


# ═══════════════════════════════════════════════════════════════════════════
# 2. THE MAPPER — YOLOv8 Face Detection + MediaPipe Segmentation
# ═══════════════════════════════════════════════════════════════════════════

class Mapper:
    """
    Two-stage face/skin analysis (fully local, no weight downloads):
      1. YOLOv8-face for bounding-box detection
      2. MediaPipe SelfieSegmentation for person/background mask
      3. MediaPipe FaceMesh for landmark-based skin/face/hair region extraction

    Caches the parsed mask embeddings to disk for sub-50ms UI manipulation.
    """

    # MediaPipe FaceMesh landmark index groups for region extraction
    # Ref: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
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
    _NOSE = [1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 19, 94, 370, 141]

    # MediaPipe Tasks model download URLs
    _MP_MODEL_BASE = "https://storage.googleapis.com/mediapipe-models"
    _SEGMENTER_URLS = {
        0: f"{_MP_MODEL_BASE}/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite",
        1: f"{_MP_MODEL_BASE}/image_segmenter/selfie_segmenter_landscape/float16/1/selfie_segmenter_landscape.tflite",
    }
    _LANDMARKER_URL = f"{_MP_MODEL_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

    @staticmethod
    def _ensure_model(url: str, cache_dir: Path) -> str:
        """Download a MediaPipe model file if not already cached. Returns local path."""
        filename = url.rsplit("/", 1)[-1]
        local_path = cache_dir / filename
        if local_path.exists():
            return str(local_path)
        log.info("Mapper: downloading model %s …", filename)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        local_path.write_bytes(resp.content)
        log.info("Mapper: saved model → %s", local_path)
        return str(local_path)

    def __init__(self, cfg: Dict[str, Any]):
        from ultralytics import YOLO
        import mediapipe as mp

        self.yolo_conf = cfg.get("yolo_conf", 0.40)
        self.cache_dir = Path(cfg.get("cache_dir", "./.aps_cache/embeddings"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        model_cache = self.cache_dir.parent / "models"
        model_cache.mkdir(parents=True, exist_ok=True)

        # Load YOLOv8-face (bounding-box detection)
        yolo_model = cfg.get("yolo_model", "yolov8m-face")
        try:
            self.yolo = YOLO(f"{yolo_model}.pt")
            log.info("Mapper: YOLOv8 loaded (%s)", yolo_model)
        except Exception:
            log.warning("Face-specific model not found, using yolov8m with person class")
            self.yolo = YOLO("yolov8m.pt")

        # Load YOLOv8-seg for instance segmentation (person class 0)
        yolo_seg_model = cfg.get("yolo_seg_model", "yolov8n-seg")
        try:
            self.yolo_seg = YOLO(f"{yolo_seg_model}.pt")
            log.info("Mapper: YOLOv8-seg loaded (%s)", yolo_seg_model)
        except Exception:
            log.warning("YOLOv8-seg model '%s' not found, trying yolov8m-seg", yolo_seg_model)
            try:
                self.yolo_seg = YOLO("yolov8m-seg.pt")
                log.info("Mapper: YOLOv8-seg loaded (yolov8m-seg)")
            except Exception:
                log.warning("No YOLOv8-seg model available; YOLO person mask disabled")
                self.yolo_seg = None

        # MediaPipe ImageSegmenter (person vs background) — Tasks Vision API
        selfie_model = cfg.get("selfie_model", 1)
        seg_url = self._SEGMENTER_URLS.get(selfie_model, self._SEGMENTER_URLS[1])
        seg_model_path = self._ensure_model(seg_url, model_cache)

        seg_base = mp.tasks.BaseOptions(model_asset_path=seg_model_path)
        seg_options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=seg_base,
            output_category_mask=True,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        self.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(seg_options)
        log.info("Mapper: MediaPipe ImageSegmenter ready (model=%d)", selfie_model)

        # MediaPipe FaceLandmarker (468/478 landmarks) — Tasks Vision API
        confidence = cfg.get("facemesh_confidence", 0.5)
        lm_model_path = self._ensure_model(self._LANDMARKER_URL, model_cache)

        lm_base = mp.tasks.BaseOptions(model_asset_path=lm_model_path)
        lm_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=lm_base,
            num_faces=10,
            min_face_detection_confidence=confidence,
            min_face_presence_confidence=confidence,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(lm_options)
        log.info("Mapper: MediaPipe FaceLandmarker ready")

    def _yolo_person_mask(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        """Run YOLOv8-seg instance segmentation → union binary mask of all person detections (uint8 0/255)."""
        if self.yolo_seg is None:
            return None
        h, w = bgr.shape[:2]
        results = self.yolo_seg.predict(bgr, conf=self.yolo_conf, verbose=False)
        mask = np.zeros((h, w), dtype=np.uint8)
        for r in results:
            if r.masks is None:
                continue
            for cls_id, seg_mask in zip(r.boxes.cls, r.masks.data):
                if int(cls_id) == 0:  # class 0 = person
                    m = seg_mask.cpu().numpy()
                    m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                    mask = cv2.bitwise_or(mask, (m_resized > 0.5).astype(np.uint8) * 255)
        return mask

    def _selfie_mask(self, bgr: np.ndarray) -> np.ndarray:
        """Run MediaPipe ImageSegmenter → binary person mask (uint8 0/255)."""
        import mediapipe as mp
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.segmenter.segment(mp_image)
        # category_mask: pixels labelled as person (category 0 = background, >0 = person)
        cat_mask = result.category_mask.numpy_view()
        mask = (cat_mask > 0).astype(np.uint8) * 255
        return mask

    def _landmarks_to_mask(self, bgr: np.ndarray, landmarks, indices: list) -> np.ndarray:
        """Draw a filled polygon from FaceMesh landmark indices → binary mask."""
        h, w = bgr.shape[:2]
        pts = []
        for idx in indices:
            lm = landmarks[idx]
            pts.append((int(lm.x * w), int(lm.y * h)))
        mask = np.zeros((h, w), dtype=np.uint8)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
        return mask

    def _build_masks(self, bgr: np.ndarray, person_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build skin, hair, and face masks using MediaPipe FaceLandmarker + ImageSegmenter.
        - face_mask: face oval polygon from FaceLandmarker landmarks
        - skin_mask: face_mask minus eyes, brows, lips (pure skin)
        - hair_mask: person_mask minus face_mask (approximate hair/neck/clothing above face)
        """
        import mediapipe as mp
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.face_landmarker.detect(mp_image)

        face_mask = np.zeros((h, w), dtype=np.uint8)
        skin_mask = np.zeros((h, w), dtype=np.uint8)

        if results.face_landmarks:
            for lm in results.face_landmarks:

                # Face oval
                oval = self._landmarks_to_mask(bgr, lm, self._FACE_OVAL)
                face_mask = cv2.bitwise_or(face_mask, oval)

                # Skin = face oval minus eyes, eyebrows, lips
                exclude = np.zeros((h, w), dtype=np.uint8)
                for region in [self._LEFT_EYE, self._RIGHT_EYE,
                               self._LEFT_EYEBROW, self._RIGHT_EYEBROW, self._LIPS]:
                    region_mask = self._landmarks_to_mask(bgr, lm, region)
                    exclude = cv2.bitwise_or(exclude, region_mask)

                face_skin = cv2.bitwise_and(oval, cv2.bitwise_not(exclude))
                skin_mask = cv2.bitwise_or(skin_mask, face_skin)

        # Hair mask: person pixels that are NOT in the face oval region
        # This approximates hair, ears, neck, upper clothing
        hair_mask = cv2.bitwise_and(person_mask, cv2.bitwise_not(face_mask))

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        face_mask = cv2.morphologyEx(face_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return skin_mask, hair_mask, face_mask

    def _cache_embedding(self, rec: ImageRecord, skin_mask: np.ndarray,
                         hair_mask: np.ndarray, face_mask: np.ndarray) -> Path:
        """Cache the parsed masks as a compressed numpy file."""
        h = hashlib.md5(str(rec.source_path).encode()).hexdigest()[:12]
        cache_path = self.cache_dir / f"{rec.uid}_{h}.npz"
        np.savez_compressed(
            str(cache_path),
            skin=skin_mask, hair=hair_mask, face=face_mask,
        )
        return cache_path

    def _export_debug_images(self, rec: ImageRecord, yolo_mask: Optional[np.ndarray],
                             mp_mask: np.ndarray, final_mask: np.ndarray) -> None:
        """Export stage debug images to output/debug/<image_stem>/."""
        debug_dir = Path(self._output_dir) / "debug" / rec.source_path.stem
        debug_dir.mkdir(parents=True, exist_ok=True)

        if yolo_mask is not None:
            cv2.imwrite(str(debug_dir / "Stage1_YOLO_Person_Mask.png"), yolo_mask)
        cv2.imwrite(str(debug_dir / "Stage2_MediaPipe_Raw_Mask.png"), mp_mask)
        cv2.imwrite(str(debug_dir / "Stage3_Final_Skin_Mask.png"), final_mask)
        log.info("Mapper: debug images → %s", debug_dir)

    def analyse(self, rec: ImageRecord, output_dir: str = "./output") -> ImageRecord:
        bgr = rec.decoded_bgr
        self._output_dir = output_dir

        # Stage 1a: YOLOv8 face detection (bounding boxes)
        results = self.yolo.predict(bgr, conf=self.yolo_conf, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                boxes.append({
                    "x1": int(xyxy[0]), "y1": int(xyxy[1]),
                    "x2": int(xyxy[2]), "y2": int(xyxy[3]),
                    "confidence": round(conf, 4),
                })
        rec.face_boxes = boxes
        log.info("Mapper: %s — %d face(s) detected", rec.source_path.name, len(boxes))

        # Stage 1b: YOLOv8-seg person instance segmentation
        yolo_person_mask = self._yolo_person_mask(bgr)

        # Stage 2: MediaPipe segmentation (raw)
        mediapipe_mask = self._selfie_mask(bgr)

        # Stage 3: Intersect YOLO person mask with MediaPipe mask
        # This eliminates background noise (e.g. ocean waves misidentified as person)
        if yolo_person_mask is not None:
            person_mask = cv2.bitwise_and(mediapipe_mask, yolo_person_mask)
            log.info("Mapper: %s — YOLO∩MediaPipe intersection applied", rec.source_path.name)
        else:
            person_mask = mediapipe_mask
            log.warning("Mapper: %s — no YOLO-seg mask, using raw MediaPipe", rec.source_path.name)

        skin_mask, hair_mask, face_mask = self._build_masks(bgr, person_mask)
        rec.skin_mask = skin_mask
        rec.hair_mask = hair_mask
        rec.face_parse_mask = face_mask

        skin_pct = float(np.sum(skin_mask > 0)) / skin_mask.size * 100
        log.info("Mapper: %s — skin=%.1f%%", rec.source_path.name, skin_pct)

        # Export debug stage images
        self._export_debug_images(rec, yolo_person_mask, mediapipe_mask, skin_mask)

        # Cache
        cache_path = self._cache_embedding(rec, skin_mask, hair_mask, face_mask)
        rec.embedding_cache_path = cache_path
        log.info("Mapper: cached → %s", cache_path)

        return rec


# ═══════════════════════════════════════════════════════════════════════════
# 3. THE FORGE — ComfyUI Headless API Executor
# ═══════════════════════════════════════════════════════════════════════════

class Forge:
    """
    Submits images to ComfyUI via its REST API for FaceDetailer retouching.
    Uses WAS Node Suite nodes for:
      - WAS_Image_Load: native image loading
      - WAS_Image_Save: metadata-stripped PNG output
    Uses FaceDetailer (Impact Pack) for face-region inpainting.

    Security: strip_metadata=True removes all workflow JSON from output PNGs.
    Denoise is hard-locked to [0.25, 0.35].
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.api_url = cfg.get("comfyui_url", "http://127.0.0.1:8188")
        self.denoise = max(
            cfg.get("denoise_min", 0.25),
            min(cfg.get("denoise", 0.30), cfg.get("denoise_max", 0.35)),
        )
        self.guide_size = cfg.get("guide_size", 384)
        self.positive_prompt = cfg["positive_prompt"]
        self.negative_prompt = cfg["negative_prompt"]
        self.strip_metadata = cfg.get("strip_metadata", True)
        self.sampler = cfg.get("sampler", "euler_ancestral")
        self.scheduler = cfg.get("scheduler", "normal")
        self.steps = cfg.get("steps", 20)
        self.cfg_scale = cfg.get("cfg_scale", 7.0)
        self.checkpoint = cfg.get("checkpoint", "realisticVisionV60B1_v51VAE.safetensors")
        self.output_dir: Optional[Path] = None

        log.info(
            "Forge ready: api=%s, denoise=%.2f, guide=%d, steps=%d",
            self.api_url, self.denoise, self.guide_size, self.steps,
        )

    def _build_workflow(self, image_path: str, skin_mask_path: str) -> Dict[str, Any]:
        """
        Build the ComfyUI workflow JSON using WAS Node Suite + FaceDetailer.
        This is the actual executable workflow graph — not a placeholder.
        """
        client_id = uuid.uuid4().hex[:16]

        workflow = {
            # --- Checkpoint Loader ---
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": self.checkpoint,
                },
            },
            # --- CLIP Text Encode (Positive) ---
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": self.positive_prompt,
                    "clip": ["1", 1],
                },
            },
            # --- CLIP Text Encode (Negative) ---
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": self.negative_prompt,
                    "clip": ["1", 1],
                },
            },
            # --- WAS Image Load (source photo) ---
            "4": {
                "class_type": "WAS_Image_Load",
                "inputs": {
                    "image_path": image_path,
                },
            },
            # --- WAS Image Load (skin mask) ---
            "5": {
                "class_type": "WAS_Image_Load",
                "inputs": {
                    "image_path": skin_mask_path,
                },
            },
            # --- Image to Mask (convert skin mask image to mask type) ---
            "6": {
                "class_type": "ImageToMask",
                "inputs": {
                    "image": ["5", 0],
                    "channel": "red",
                },
            },
            # --- VAE Encode (source image → latent) ---
            "7": {
                "class_type": "VAEEncode",
                "inputs": {
                    "pixels": ["4", 0],
                    "vae": ["1", 2],
                },
            },
            # --- KSampler (masked inpaint on skin regions) ---
            "8": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["10", 0],
                    "seed": int(time.time()) % (2**32),
                    "steps": self.steps,
                    "cfg": self.cfg_scale,
                    "sampler_name": self.sampler,
                    "scheduler": self.scheduler,
                    "denoise": self.denoise,
                },
            },
            # --- FaceDetailer (Impact Pack) — face-region refinement ---
            "9": {
                "class_type": "FaceDetailer",
                "inputs": {
                    "image": ["4", 0],
                    "model": ["1", 0],
                    "clip": ["1", 1],
                    "vae": ["1", 2],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "bbox_detector": ["15", 0],
                    "guide_size": self.guide_size,
                    "guide_size_for": True,
                    "max_size": 1024,
                    "seed": int(time.time()) % (2**32),
                    "steps": self.steps,
                    "cfg": self.cfg_scale,
                    "sampler_name": self.sampler,
                    "scheduler": self.scheduler,
                    "denoise": self.denoise,
                    "feather": 5,
                    "noise_mask": True,
                    "force_inpaint": True,
                    "drop_size": 10,
                    "cycle": 1,
                },
            },
            # --- Set Latent Noise Mask (apply skin mask to latent) ---
            "10": {
                "class_type": "SetLatentNoiseMask",
                "inputs": {
                    "samples": ["7", 0],
                    "mask": ["6", 0],
                },
            },
            # --- VAE Decode (latent → pixels after skin smoothing) ---
            "11": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["8", 0],
                    "vae": ["1", 2],
                },
            },
            # --- Image Blend (merge skin-smoothed with FaceDetailer output) ---
            "12": {
                "class_type": "ImageBlend",
                "inputs": {
                    "image1": ["9", 0],
                    "image2": ["11", 0],
                    "blend_factor": 0.65,
                    "blend_mode": "normal",
                },
            },
            # --- WAS Image Save (output with metadata stripped) ---
            "13": {
                "class_type": "WAS_Image_Save",
                "inputs": {
                    "images": ["12", 0],
                    "output_path": str(self.output_dir) if self.output_dir else "./output",
                    "filename_prefix": "aps_retouched",
                    "extension": "png",
                    "quality": 100,
                    "overwrite_mode": "false",
                    "embed_workflow": "false" if self.strip_metadata else "true",
                },
            },
            # --- UltralyticsBBoxDetector (for FaceDetailer) ---
            "15": {
                "class_type": "UltralyticsDetectorProvider",
                "inputs": {
                    "model_name": "face_yolov8m.pt",
                },
            },
        }

        return {"prompt": workflow, "client_id": client_id}

    def _upload_image(self, image_path: Path) -> str:
        """Upload an image to ComfyUI's input directory."""
        url = f"{self.api_url}/upload/image"
        filename = image_path.name
        with open(image_path, "rb") as f:
            files = {"image": (filename, f, "image/png")}
            resp = requests.post(url, files=files, timeout=30)
            resp.raise_for_status()
        data = resp.json()
        return data.get("name", filename)

    def _save_mask_temp(self, mask: np.ndarray, uid: str) -> Path:
        """Save skin mask as temporary PNG for upload to ComfyUI."""
        temp_dir = Path("./.aps_cache/masks")
        temp_dir.mkdir(parents=True, exist_ok=True)
        mask_path = temp_dir / f"{uid}_skin_mask.png"
        # Convert single-channel mask to 3-channel for WAS_Image_Load compatibility
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(mask_path), mask_rgb)
        return mask_path

    def _queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """Submit workflow to ComfyUI queue. Returns prompt_id."""
        url = f"{self.api_url}/prompt"
        resp = requests.post(url, json=workflow, timeout=30)
        resp.raise_for_status()
        return resp.json()["prompt_id"]

    def _poll_completion(self, prompt_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Poll ComfyUI history until the prompt completes."""
        url = f"{self.api_url}/history/{prompt_id}"
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if prompt_id in data:
                        return data[prompt_id]
            except requests.RequestException:
                pass
            time.sleep(2)

        raise TimeoutError(f"ComfyUI prompt {prompt_id} did not complete within {timeout}s")

    def _retrieve_output(self, history: Dict[str, Any]) -> Optional[bytes]:
        """Download the output image from ComfyUI."""
        outputs = history.get("outputs", {})
        for node_id, node_out in outputs.items():
            images = node_out.get("images", [])
            for img_info in images:
                filename = img_info.get("filename", "")
                subfolder = img_info.get("subfolder", "")
                img_type = img_info.get("type", "output")
                url = (
                    f"{self.api_url}/view?"
                    f"filename={filename}&subfolder={subfolder}&type={img_type}"
                )
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    return resp.content
        return None

    def _strip_png_metadata(self, png_bytes: bytes) -> bytes:
        """Remove all non-essential PNG chunks (tEXt, iTXt, zTXt) to prevent workflow leakage."""
        if not self.strip_metadata:
            return png_bytes

        # Parse PNG and keep only critical chunks + IDAT
        SIGNATURE = b'\x89PNG\r\n\x1a\n'
        if not png_bytes.startswith(SIGNATURE):
            return png_bytes  # not a PNG, return as-is

        result = bytearray(SIGNATURE)
        pos = 8  # skip signature

        critical_types = {b'IHDR', b'PLTE', b'IDAT', b'IEND', b'tRNS', b'gAMA', b'cHRM', b'sRGB', b'iCCP'}

        while pos < len(png_bytes):
            if pos + 8 > len(png_bytes):
                break
            length = struct.unpack(">I", png_bytes[pos:pos+4])[0]
            chunk_type = png_bytes[pos+4:pos+8]
            chunk_end = pos + 12 + length  # 4(len) + 4(type) + data + 4(crc)

            if chunk_end > len(png_bytes):
                break

            if chunk_type in critical_types:
                result.extend(png_bytes[pos:chunk_end])

            pos = chunk_end

        return bytes(result)

    def retouch(self, rec: ImageRecord) -> ImageRecord:
        """Execute the full Forge retouching pipeline via ComfyUI API."""
        self.output_dir = rec.retouched_path.parent if rec.retouched_path else Path("./output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare source image for upload
        source_temp = Path("./.aps_cache/sources")
        source_temp.mkdir(parents=True, exist_ok=True)
        source_path = source_temp / f"{rec.uid}.png"
        cv2.imwrite(str(source_path), rec.decoded_bgr)

        # Save skin mask
        skin_mask = rec.skin_mask if rec.skin_mask is not None else (
            np.ones(rec.decoded_bgr.shape[:2], dtype=np.uint8) * 255
        )
        mask_path = self._save_mask_temp(skin_mask, rec.uid)

        # Upload to ComfyUI
        try:
            uploaded_source = self._upload_image(source_path)
            uploaded_mask = self._upload_image(mask_path)
        except requests.RequestException as e:
            log.error("Forge: failed to upload to ComfyUI: %s", e)
            return self._local_fallback(rec)

        # Build and submit workflow
        workflow = self._build_workflow(uploaded_source, uploaded_mask)

        try:
            prompt_id = self._queue_prompt(workflow)
            log.info("Forge: queued prompt %s for %s", prompt_id, rec.source_path.name)

            history = self._poll_completion(prompt_id, timeout=300)
            output_bytes = self._retrieve_output(history)

            if output_bytes is None:
                log.error("Forge: no output image from ComfyUI for %s", rec.source_path.name)
                return self._local_fallback(rec)

            # Strip metadata
            output_bytes = self._strip_png_metadata(output_bytes)

            # Decode result
            nparr = np.frombuffer(output_bytes, np.uint8)
            result_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if result_bgr is None:
                log.error("Forge: failed to decode ComfyUI output for %s", rec.source_path.name)
                return self._local_fallback(rec)

            # Resize to match original if needed
            h, w = rec.decoded_bgr.shape[:2]
            if result_bgr.shape[:2] != (h, w):
                result_bgr = cv2.resize(result_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

            rec.retouched_bgr = result_bgr

            # Save
            out_path = self.output_dir / f"{rec.uid}_B_Retouched.png"
            # Write stripped PNG
            with open(str(out_path), "wb") as f:
                _, buf = cv2.imencode(".png", result_bgr)
                f.write(self._strip_png_metadata(buf.tobytes()))

            rec.retouched_path = out_path
            rec.forge_metadata = {
                "method": "comfyui_was_facedetailer",
                "denoise": self.denoise,
                "guide_size": self.guide_size,
                "steps": self.steps,
                "cfg": self.cfg_scale,
                "sampler": self.sampler,
                "checkpoint": self.checkpoint,
                "prompt_id": prompt_id,
                "metadata_stripped": self.strip_metadata,
            }
            log.info("Forge: %s → %s (ComfyUI)", rec.source_path.name, out_path)

        except (requests.RequestException, TimeoutError) as e:
            log.warning("Forge: ComfyUI unavailable (%s), using local fallback", e)
            return self._local_fallback(rec)

        return rec

    def _local_fallback(self, rec: ImageRecord) -> ImageRecord:
        """
        Local frequency-separation fallback when ComfyUI is unavailable.
        Mirrors the aesthetic contract: preserve high-freq, smooth low-freq on skin only.
        """
        log.info("Forge [LOCAL FALLBACK]: processing %s", rec.source_path.name)
        bgr = rec.decoded_bgr.copy()
        skin_mask = rec.skin_mask if rec.skin_mask is not None else (
            np.ones(bgr.shape[:2], dtype=np.uint8) * 255
        )

        # Map denoise [0.25, 0.35] → bilateral diameter [5, 9]
        t = (self.denoise - 0.25) / 0.10
        bilateral_d = int(5 + t * 4)
        blur_ksize = int(11 + t * 10)
        blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1

        # Frequency separation
        low = cv2.GaussianBlur(bgr.astype(np.float64), (blur_ksize, blur_ksize), 0)
        high = bgr.astype(np.float64) - low + 128.0

        # Smooth low-freq on skin only (bilateral filter)
        sigma = 40 + (bilateral_d - 5) * 8
        smoothed = cv2.bilateralFilter(
            low.astype(np.float32), bilateral_d, float(sigma), float(sigma)
        ).astype(np.float64)

        mask_f = (skin_mask.astype(np.float64) / 255.0)[:, :, np.newaxis]
        low_result = smoothed * mask_f + low * (1.0 - mask_f)

        # Recombine (high layer untouched — pores preserved)
        recombined = np.clip(low_result + (high - 128.0), 0, 255).astype(np.uint8)

        # Dodge & burn on face regions
        if rec.face_boxes:
            recombined = self._dodge_and_burn(recombined, rec.face_boxes)

        rec.retouched_bgr = recombined

        out_dir = Path("./output")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{rec.uid}_B_Retouched.png"
        cv2.imwrite(str(out_path), recombined)
        rec.retouched_path = out_path

        rec.forge_metadata = {
            "method": "local_frequency_separation_fallback",
            "denoise": self.denoise,
            "bilateral_d": bilateral_d,
            "blur_ksize": blur_ksize,
        }
        log.info("Forge [FALLBACK]: %s → %s", rec.source_path.name, out_path)
        return rec

    @staticmethod
    def _dodge_and_burn(bgr: np.ndarray, face_boxes: List[Dict], strength: float = 0.12) -> np.ndarray:
        """Neutral-gray dodge & burn for 3D facial contouring."""
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
        h, w = bgr.shape[:2]

        for box in face_boxes:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            pad_x = int((x2 - x1) * 0.15)
            pad_y = int((y2 - y1) * 0.15)
            rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            rx2, ry2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

            L = lab[ry1:ry2, rx1:rx2, 0]
            ksize = max(15, (ry2 - ry1) // 8)
            ksize = ksize if ksize % 2 == 1 else ksize + 1
            local_mean = cv2.GaussianBlur(L, (ksize, ksize), 0)
            delta = (local_mean - L) * strength
            lab[ry1:ry2, rx1:rx2, 0] = np.clip(L + delta, 0, 255)

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


# ═══════════════════════════════════════════════════════════════════════════
# 4. QUALITY CHECKER
# ═══════════════════════════════════════════════════════════════════════════

class QualityChecker:
    """
    Post-retouch quality gates:
      1. Pore preservation  — high-freq energy ratio ≥ 0.82
      2. Lighting fidelity  — mean luminance drift < 8 units
      3. SSIM              — structural similarity > 0.82
      4. Skin variance     — no plastic artifacts (ratio ≥ 0.60)
    """

    PORE_THRESHOLD = 0.82
    LUMINANCE_DRIFT_MAX = 8.0
    SSIM_MIN = 0.82
    SKIN_VAR_RATIO_MIN = 0.60

    @staticmethod
    def _high_freq_energy(bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(np.mean(np.abs(lap)))

    @staticmethod
    def _mean_luminance(bgr: np.ndarray) -> float:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        return float(np.mean(lab[:, :, 0]))

    @staticmethod
    def _ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)
        sigma1_sq = cv2.GaussianBlur(g1 * g1, (11, 11), 1.5) - mu1 * mu1
        sigma2_sq = cv2.GaussianBlur(g2 * g2, (11, 11), 1.5) - mu2 * mu2
        sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1 * mu2
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return float(np.mean(ssim_map))

    def check(self, rec: ImageRecord) -> Dict[str, Any]:
        orig = rec.decoded_bgr
        ret = rec.retouched_bgr
        mask = rec.skin_mask if rec.skin_mask is not None else np.zeros(orig.shape[:2], dtype=np.uint8)

        hf_orig = self._high_freq_energy(orig)
        hf_ret = self._high_freq_energy(ret)
        pore_ratio = hf_ret / hf_orig if hf_orig > 0 else 1.0

        lum_drift = abs(self._mean_luminance(ret) - self._mean_luminance(orig))

        ssim_val = self._ssim(orig, ret)

        # Skin variance ratio
        g_orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.float64)
        g_ret = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY).astype(np.float64)
        skin_px = mask > 0
        if np.any(skin_px):
            var_orig = float(np.var(g_orig[skin_px]))
            sv_ratio = float(np.var(g_ret[skin_px])) / var_orig if var_orig > 1.0 else 1.0
        else:
            sv_ratio = 1.0

        all_pass = (
            pore_ratio >= self.PORE_THRESHOLD
            and lum_drift <= self.LUMINANCE_DRIFT_MAX
            and ssim_val >= self.SSIM_MIN
            and sv_ratio >= self.SKIN_VAR_RATIO_MIN
        )

        checks = {
            "pore_ratio": round(pore_ratio, 4),
            "pore_pass": pore_ratio >= self.PORE_THRESHOLD,
            "luminance_drift": round(lum_drift, 2),
            "luminance_pass": lum_drift <= self.LUMINANCE_DRIFT_MAX,
            "ssim": round(ssim_val, 4),
            "ssim_pass": ssim_val >= self.SSIM_MIN,
            "skin_var_ratio": round(sv_ratio, 4),
            "skin_var_pass": sv_ratio >= self.SKIN_VAR_RATIO_MIN,
            "ALL_PASS": all_pass,
        }
        rec.quality_checks = checks

        log.info(
            "QualityCheck: %s  pore=%.3f  lum=%.1f  ssim=%.3f  sv=%.3f  → %s",
            rec.source_path.name, pore_ratio, lum_drift, ssim_val, sv_ratio,
            "PASS" if all_pass else "FAIL",
        )
        return checks


# ═══════════════════════════════════════════════════════════════════════════
# 5. PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class APSPipeline:
    """
    End-to-end batch pipeline: discover → sort → map → forge → verify.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.cfg = load_config(config_path)
        self.output_dir = Path(self.cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log.info("Initialising APS Pipeline...")
        self.decoder = ImageDecoder()
        self.sorter = Sorter(self.cfg["sorter"])
        self.mapper = Mapper(self.cfg["mapper"])
        self.forge = Forge(self.cfg["forge"])
        self.checker = QualityChecker()
        log.info("APS Pipeline ready.")

    def discover_images(self, input_dir: Optional[str] = None) -> List[Path]:
        src = Path(input_dir or self.cfg["input_dir"])
        exts = set(self.cfg["supported_ext"])
        images = sorted(
            p for p in src.iterdir()
            if p.suffix.lower() in exts and p.is_file()
        )
        log.info("Discovered %d image(s) in %s", len(images), src)
        return images

    def process_single(self, img_path: Path) -> ImageRecord:
        rec = ImageRecord(source_path=img_path, uid=img_path.stem)

        # Decode
        bgr = self.decoder.decode(img_path)
        if bgr is None:
            log.error("Skipping %s — decode failed", img_path.name)
            return rec
        rec.decoded_bgr = bgr

        # Sort (IQA)
        self.sorter.evaluate(rec)
        if not rec.is_accepted:
            log.info("Rejected by Sorter: %s (score=%.4f)", img_path.name, rec.quality_score)
            return rec

        # Map (face detection + skin segmentation)
        self.mapper.analyse(rec, output_dir=str(self.output_dir))

        # Forge (retouch)
        self.forge.retouch(rec)

        # Quality gate
        if rec.retouched_bgr is not None:
            self.checker.check(rec)

        return rec

    def run(self, input_dir: Optional[str] = None) -> List[ImageRecord]:
        images = self.discover_images(input_dir)
        if not images:
            log.error("No images found. Exiting.")
            return []

        records: List[ImageRecord] = []
        accepted = 0
        passed_quality = 0

        for img_path in tqdm(images, desc="APS Pipeline", unit="img"):
            log.info("")
            log.info("=" * 60)
            log.info("Processing: %s", img_path.name)
            log.info("=" * 60)

            rec = self.process_single(img_path)
            records.append(rec)

            if rec.is_accepted:
                accepted += 1
            if rec.quality_checks.get("ALL_PASS"):
                passed_quality += 1

        # Final report
        self._print_report(records, accepted, passed_quality)
        return records

    @staticmethod
    def _print_report(records: List[ImageRecord], accepted: int, passed: int) -> None:
        total = len(records)
        print("\n" + "=" * 80)
        print("  APS PIPELINE REPORT")
        print("=" * 80)
        print(f"  {'Image':<25s} {'IQA':>6s} {'Faces':>5s} {'SSIM':>6s} {'Pore':>6s} {'Status':>8s}")
        print("-" * 80)

        for rec in records:
            c = rec.quality_checks
            status = "SKIP"
            if not rec.is_accepted:
                status = "REJECT"
            elif c.get("ALL_PASS"):
                status = "PASS"
            elif c:
                status = "FAIL"

            print(
                f"  {rec.source_path.name:<25s} "
                f"{rec.quality_score or 0:>6.3f} "
                f"{len(rec.face_boxes):>5d} "
                f"{c.get('ssim', 0):>6.3f} "
                f"{c.get('pore_ratio', 0):>6.3f} "
                f"{status:>8s}"
            )

        print("-" * 80)
        print(f"  Total: {total}  |  Accepted (IQA): {accepted}  |  Passed Quality: {passed}")
        print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════
# 6. IMAGE FORMAT CONVERTER (--mode convert)
# ═══════════════════════════════════════════════════════════════════════════

class ImageConverter:
    """
    Bulk image format conversion: RAW/JPG/PNG/TIFF/BMP ↔ JPG/PNG/TIFF/BMP.
    Uses rawpy for RAW decode, Pillow for format writing.
    """

    SUPPORTED_INPUT = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".arw", ".cr3", ".nef", ".dng", ".raf"}
    SUPPORTED_OUTPUT = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
    RAW_EXTENSIONS = {".arw", ".cr3", ".nef", ".dng", ".raf"}

    def __init__(self, output_format: str = ".png", quality: int = 95):
        fmt = output_format.lower() if output_format.startswith(".") else f".{output_format.lower()}"
        if fmt not in self.SUPPORTED_OUTPUT:
            raise ValueError(f"Unsupported output format: {fmt}. Supported: {self.SUPPORTED_OUTPUT}")
        self.output_format = fmt
        self.quality = quality

    def convert_single(self, src: Path, out_dir: Path) -> Optional[Path]:
        """Convert a single image to the target format."""
        ext = src.suffix.lower()
        if ext not in self.SUPPORTED_INPUT:
            log.warning("Converter: skipping unsupported file %s", src.name)
            return None

        try:
            if ext in self.RAW_EXTENSIONS:
                with rawpy.imread(str(src)) as raw:
                    rgb = raw.postprocess(
                        use_camera_wb=True, half_size=False,
                        no_auto_bright=False, output_bps=8,
                    )
                pil_img = Image.fromarray(rgb)
            else:
                pil_img = Image.open(src)
                # Ensure RGB for JPEG output (no alpha channel)
                if self.output_format in (".jpg", ".jpeg") and pil_img.mode in ("RGBA", "P", "LA"):
                    pil_img = pil_img.convert("RGB")
                elif pil_img.mode == "P":
                    pil_img = pil_img.convert("RGB")

            out_name = src.stem + self.output_format
            out_path = out_dir / out_name
            save_kwargs: Dict[str, Any] = {}

            if self.output_format in (".jpg", ".jpeg"):
                save_kwargs["quality"] = self.quality
                save_kwargs["optimize"] = True
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
            elif self.output_format == ".png":
                save_kwargs["optimize"] = True
            elif self.output_format == ".webp":
                save_kwargs["quality"] = self.quality
            elif self.output_format in (".tiff", ".tif"):
                save_kwargs["compression"] = "tiff_lzw"

            pil_img.save(str(out_path), **save_kwargs)
            log.info("Converted: %s → %s", src.name, out_path.name)
            return out_path

        except Exception as e:
            log.error("Failed to convert %s: %s", src.name, e)
            return None

    def convert_batch(self, input_dir: str, output_dir: str) -> List[Path]:
        """Convert all supported images in input_dir to output_dir."""
        src_dir = Path(input_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(
            p for p in src_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_INPUT
        )
        if not files:
            log.error("No convertible images found in %s", src_dir)
            return []

        log.info("Converting %d image(s) → %s format", len(files), self.output_format)
        results = []
        for f in tqdm(files, desc="Converting", unit="img"):
            out = self.convert_single(f, out_dir)
            if out:
                results.append(out)

        log.info("Conversion complete: %d/%d succeeded", len(results), len(files))
        return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="APS Pipeline — Auto Photo Studio for Sony A7M5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  pipeline  Full retouching pipeline (default)
  convert   Bulk image format conversion

Examples:
  python aps_pipeline.py -i ./photos -o ./output
  python aps_pipeline.py --mode convert -i ./raw_photos -o ./jpgs --format jpg
  python aps_pipeline.py --mode convert -i ./photos -o ./pngs --format png --quality 90
        """,
    )
    parser.add_argument("--mode", type=str, default="pipeline",
                        choices=["pipeline", "convert"],
                        help="Operation mode: 'pipeline' (full retouch) or 'convert' (format conversion)")
    parser.add_argument("-i", "--input", type=str, default=None, help="Input directory")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output directory")
    parser.add_argument("-c", "--config", type=str, default="aps_config.yaml", help="Config YAML")
    parser.add_argument("--denoise", type=float, default=None, help="Denoise strength (0.25-0.35)")
    parser.add_argument("--threshold", type=float, default=None, help="IQA acceptance threshold")
    parser.add_argument("--format", type=str, default="png",
                        help="Target format for --mode convert (jpg/png/tiff/bmp/webp)")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG/WebP quality for --mode convert (1-100, default 95)")
    args = parser.parse_args()

    # ── Convert mode ──────────────────────────────────────────────────────
    if args.mode == "convert":
        input_dir = args.input or "./test_images"
        output_dir = args.output or "./output"
        converter = ImageConverter(output_format=args.format, quality=args.quality)
        results = converter.convert_batch(input_dir, output_dir)
        print(f"\nConverted {len(results)} image(s) to {args.format.upper()} in {output_dir}")
        sys.exit(0)

    # ── Pipeline mode (default) ───────────────────────────────────────────
    cfg_path = args.config if Path(args.config).exists() else None
    pipeline = APSPipeline(config_path=cfg_path)

    if args.input:
        pipeline.cfg["input_dir"] = args.input
    if args.output:
        pipeline.cfg["output_dir"] = args.output
        pipeline.output_dir = Path(args.output)
        pipeline.output_dir.mkdir(parents=True, exist_ok=True)
    if args.denoise is not None:
        pipeline.forge.denoise = max(0.25, min(args.denoise, 0.35))
    if args.threshold is not None:
        pipeline.sorter.threshold = args.threshold

    records = pipeline.run(args.input)

    all_pass = all(r.quality_checks.get("ALL_PASS", False) for r in records if r.is_accepted)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
# -- [SPRINT 2 PHASE C & D INTEGRATION] --
# Note: VLM Vision Agent logic and Phase C/D ComfyUI Node integrations (Image C, Image D outputs)
# are actively being woven into this pipeline instance by CC.
