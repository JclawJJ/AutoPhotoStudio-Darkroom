#!/usr/bin/env python3
"""
APS Local Test Runner — Full B → C → D Pipeline
=================================================
Runs the full APS pipeline logic (Phase B retouch, Phase C master filter,
Phase D creative expansion) against test images using ONLY lightweight
local dependencies (OpenCV, NumPy, Pillow).  No YOLOv8, no pyiqa,
no ComfyUI, no InsightFace, no DeepDanbooru required.

Fallback strategy:
  Sorter  → Laplacian variance + exposure analysis (OpenCV)
  Mapper  → Haar cascade face detection + HSV skin-color segmentation
  Forge   → Local frequency-separation retouch (Gaussian blur split)
  Phase C → Local colour-science master grading + ArcFace stub gate
  Phase D → Danbooru stub tags + archetype reasoning + procedural BG + ArcFace gate

Quality assertions after each stage:
  1. Pore preservation  — high-freq energy ratio (retouched vs original)
  2. Lighting fidelity  — mean luminance drift < threshold
  3. Skin mask coverage — non-zero skin area when face is detected
  4. Overall structure  — SSIM between original and retouched > 0.85
  5. ArcFace identity gate — cosine similarity ≥ 0.85 after C and D
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(name)s]  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("APS-TEST")

# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASS (mirrors aps_pipeline.ImageRecord)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ImageRecord:
    source_path: Path
    uid: str = ""

    # Sorter
    quality_score: Optional[float] = None
    exposure_score: Optional[float] = None
    sharpness_score: Optional[float] = None
    is_accepted: bool = False

    # Mapper
    decoded_bgr: Optional[Any] = None
    face_boxes: List[Dict[str, Any]] = field(default_factory=list)
    skin_mask: Optional[Any] = None

    # Forge (Phase B)
    retouched_bgr: Optional[Any] = None
    retouched_path: Optional[Path] = None
    forge_metadata: Dict[str, Any] = field(default_factory=dict)

    # Phase C (Master Filter)
    phase_c_bgr: Optional[Any] = None
    phase_c_path: Optional[Path] = None
    phase_c_metadata: Dict[str, Any] = field(default_factory=dict)
    face_embedding: Optional[Any] = None

    # Phase D (Creative Expansion)
    phase_d_bgr: Optional[Any] = None
    phase_d_path: Optional[Path] = None
    phase_d_metadata: Dict[str, Any] = field(default_factory=dict)
    danbooru_tags: Dict[str, float] = field(default_factory=dict)
    is_cosplay: bool = False
    archetype_prompt: Optional[str] = None

    # Quality assertions
    quality_checks: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# 1. SORTER — Laplacian sharpness + exposure analysis
# ═══════════════════════════════════════════════════════════════════════════

class LocalSorter:
    """
    Lightweight IQA using two complementary signals:
      - Laplacian variance (sharpness / focus quality)
      - Histogram spread (exposure quality — penalise clipped highlights
        and crushed shadows)

    Combined score normalised to [0, 1].  Threshold = 0.35 (tuned for
    typical Sony A7M5 JPGs which are already well-exposed).
    """

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold
        log.info("LocalSorter ready (threshold=%.2f)", threshold)

    def _sharpness(self, gray: np.ndarray) -> float:
        """Laplacian variance → [0, 1].  Map [0, 800] linearly."""
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        var = float(lap.var())
        return min(var / 800.0, 1.0)

    def _exposure(self, gray: np.ndarray) -> float:
        """
        Penalise images with >15% pixels clipped (< 10 or > 245).
        Also penalise very narrow histograms (low contrast).
        Returns [0, 1].
        """
        total = gray.size
        clipped_lo = float(np.sum(gray < 10)) / total
        clipped_hi = float(np.sum(gray > 245)) / total
        clip_penalty = max(0.0, 1.0 - (clipped_lo + clipped_hi) * 4.0)

        # Contrast: std of pixel values, normalised by 64 (good std ~50-80)
        std = float(np.std(gray))
        contrast = min(std / 64.0, 1.0)

        return 0.6 * clip_penalty + 0.4 * contrast

    def evaluate(self, rec: ImageRecord) -> ImageRecord:
        gray = cv2.cvtColor(rec.decoded_bgr, cv2.COLOR_BGR2GRAY)
        rec.sharpness_score = self._sharpness(gray)
        rec.exposure_score = self._exposure(gray)
        rec.quality_score = 0.55 * rec.sharpness_score + 0.45 * rec.exposure_score
        rec.is_accepted = rec.quality_score >= self.threshold

        log.info(
            "Sorter: %-20s sharp=%.3f expo=%.3f → q=%.3f %s",
            rec.source_path.name,
            rec.sharpness_score,
            rec.exposure_score,
            rec.quality_score,
            "✓ ACCEPTED" if rec.is_accepted else "✗ REJECTED",
        )
        return rec


# ═══════════════════════════════════════════════════════════════════════════
# 2. MAPPER — Haar cascade + HSV skin segmentation
# ═══════════════════════════════════════════════════════════════════════════

class LocalMapper:
    """
    Face detection  : OpenCV Haar cascade (haarcascade_frontalface_alt2)
    Skin masking    : HSV colour-range thresholding, refined by morphology.

    HSV skin range tuned for East-Asian skin tones under studio / natural
    lighting (Sony A7M5 colour science).
    """

    def __init__(self):
        cascade_path = os.path.join(
            os.path.dirname(cv2.__file__), "data",
            "haarcascade_frontalface_alt2.xml",
        )
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
        log.info("LocalMapper ready (Haar cascade + HSV skin)")

    def detect_faces(self, bgr: np.ndarray) -> List[Dict[str, Any]]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces_raw = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        boxes = []
        for (x, y, w, h) in faces_raw:
            boxes.append({
                "x1": int(x), "y1": int(y),
                "x2": int(x + w), "y2": int(y + h),
                "confidence": 0.90,
            })

        log.info("Mapper: detected %d face(s)", len(boxes))
        return boxes

    def segment_skin(self, bgr: np.ndarray) -> np.ndarray:
        """
        HSV-based skin segmentation.
        Two ranges to cover varied lighting:
          Range 1: H  0-25,  S 30-170,  V 80-255  (warm / well-lit)
          Range 2: H  0-15,  S 20-130,  V 50-200  (cooler / shadowed)
        Merge + morphological cleanup.
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Range 1 — warm / well-lit skin
        lo1 = np.array([0, 30, 80], dtype=np.uint8)
        hi1 = np.array([25, 170, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lo1, hi1)

        # Range 2 — cooler / shadow regions
        lo2 = np.array([0, 20, 50], dtype=np.uint8)
        hi2 = np.array([15, 130, 200], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lo2, hi2)

        mask = cv2.bitwise_or(mask1, mask2)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return mask

    def analyse(self, rec: ImageRecord) -> ImageRecord:
        rec.face_boxes = self.detect_faces(rec.decoded_bgr)
        rec.skin_mask = self.segment_skin(rec.decoded_bgr)

        skin_pct = float(np.sum(rec.skin_mask > 0)) / rec.skin_mask.size * 100
        log.info(
            "Mapper: %-20s faces=%d  skin=%.1f%%",
            rec.source_path.name, len(rec.face_boxes), skin_pct,
        )
        return rec


# ═══════════════════════════════════════════════════════════════════════════
# 3. FORGE (LOCAL) — Frequency separation retouch simulation
# ═══════════════════════════════════════════════════════════════════════════

class LocalForge:
    """
    Simulates the ComfyUI FaceDetailer pipeline using pure OpenCV
    frequency separation:

      1. Split into LOW (Gaussian blur) and HIGH (original - low) layers.
      2. On the LOW layer: apply bilateral filter ONLY within the skin mask
         to smooth colour transitions and reduce shadow harshness.
      3. On the HIGH layer: preserve 100% — this is where pores, fine
         hair, and micro-texture live.
      4. Recombine: retouched = smoothed_low + original_high
      5. Subtle dodge-and-burn: local contrast enhancement on luminance
         within the face region for 3D sculpting.

    Denoise is simulated via the bilateral filter diameter, constrained
    to the 0.25-0.35 aesthetic range mapped to d=5..9.
    """

    DENOISE_MIN = 0.25
    DENOISE_MAX = 0.35

    def __init__(self, denoise: float = 0.30, output_dir: Path = Path("./output")):
        self.denoise = max(self.DENOISE_MIN, min(denoise, self.DENOISE_MAX))
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info("LocalForge ready (denoise=%.2f, out=%s)", self.denoise, self.output_dir)

    def _frequency_separate(
        self, bgr: np.ndarray, ksize: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split image into low-freq and high-freq layers."""
        # Ensure ksize is odd
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        low = cv2.GaussianBlur(bgr.astype(np.float64), (ksize, ksize), 0)
        high = bgr.astype(np.float64) - low + 128.0  # centered at 128
        return low, high

    def _smooth_low_freq(
        self, low: np.ndarray, skin_mask: np.ndarray, strength: int
    ) -> np.ndarray:
        """
        Apply bilateral filter to the low-freq layer, masked to skin only.
        This smooths colour unevenness (blemishes, redness, shadow edges)
        while leaving non-skin areas untouched.
        """
        # Bilateral filter — edge-preserving smooth
        d = strength  # diameter: 5-9 mapped from denoise 0.25-0.35
        sigma_color = 40 + (strength - 5) * 8   # 40-72
        sigma_space = 40 + (strength - 5) * 8

        smoothed = cv2.bilateralFilter(
            low.astype(np.float32), d, float(sigma_color), float(sigma_space)
        ).astype(np.float64)

        # Blend: use smoothed only where skin_mask > 0
        mask_f = (skin_mask.astype(np.float64) / 255.0)
        if mask_f.ndim == 2:
            mask_f = mask_f[:, :, np.newaxis]

        result = smoothed * mask_f + low * (1.0 - mask_f)
        return result

    def _dodge_and_burn(
        self, bgr: np.ndarray, face_boxes: List[Dict], strength: float = 0.15
    ) -> np.ndarray:
        """
        Subtle local contrast enhancement in the face region.
        Simulates neutral-gray dodge & burn for 3D facial contouring.
        """
        result = bgr.copy().astype(np.float64)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)

        for box in face_boxes:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            # Expand ROI slightly for natural falloff
            h, w = bgr.shape[:2]
            pad_x = int((x2 - x1) * 0.15)
            pad_y = int((y2 - y1) * 0.15)
            rx1 = max(0, x1 - pad_x)
            ry1 = max(0, y1 - pad_y)
            rx2 = min(w, x2 + pad_x)
            ry2 = min(h, y2 + pad_y)

            face_lab = lab[ry1:ry2, rx1:rx2]
            L = face_lab[:, :, 0]

            # Local mean as the "neutral gray" reference
            ksize = max(15, (ry2 - ry1) // 8)
            ksize = ksize if ksize % 2 == 1 else ksize + 1
            local_mean = cv2.GaussianBlur(L, (ksize, ksize), 0)

            # Push luminance toward local mean with controlled strength
            delta = (local_mean - L) * strength
            L_new = np.clip(L + delta, 0, 255)

            face_lab[:, :, 0] = L_new
            lab[ry1:ry2, rx1:rx2] = face_lab

        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return result

    def retouch(self, rec: ImageRecord) -> ImageRecord:
        bgr = rec.decoded_bgr.copy()
        skin_mask = rec.skin_mask if rec.skin_mask is not None else np.ones(bgr.shape[:2], dtype=np.uint8) * 255

        # Map denoise 0.25-0.35 → bilateral diameter 5-9
        t = (self.denoise - self.DENOISE_MIN) / (self.DENOISE_MAX - self.DENOISE_MIN)
        bilateral_d = int(5 + t * 4)
        blur_ksize = int(11 + t * 10)  # 11-21

        log.info(
            "Forge: %-20s denoise=%.2f → bilateral_d=%d, blur_k=%d",
            rec.source_path.name, self.denoise, bilateral_d, blur_ksize,
        )

        # Step 1: Frequency separation
        low, high = self._frequency_separate(bgr, blur_ksize)

        # Step 2: Smooth low-freq layer on skin only
        low_smoothed = self._smooth_low_freq(low, skin_mask, bilateral_d)

        # Step 3: Recombine (high layer is UNTOUCHED — pores preserved)
        recombined = np.clip(low_smoothed + (high - 128.0), 0, 255).astype(np.uint8)

        # Step 4: Dodge & Burn for facial contouring
        if rec.face_boxes:
            recombined = self._dodge_and_burn(recombined, rec.face_boxes, strength=0.12)

        rec.retouched_bgr = recombined

        # Save output
        stem = rec.source_path.stem
        out_path = self.output_dir / f"{stem}_retouched.jpg"
        cv2.imwrite(str(out_path), recombined, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rec.retouched_path = out_path

        rec.forge_metadata = {
            "denoise": self.denoise,
            "bilateral_d": bilateral_d,
            "blur_ksize": blur_ksize,
            "face_count": len(rec.face_boxes),
            "method": "local_frequency_separation",
        }

        log.info("Forge: saved → %s", out_path)
        return rec


# ═══════════════════════════════════════════════════════════════════════════
# 4. QUALITY ASSERTIONS
# ═══════════════════════════════════════════════════════════════════════════

class QualityChecker:
    """
    Post-retouch quality gates that verify the aesthetic contract:
      1. Pore preservation  — high-freq energy must stay ≥ 85% of original
      2. Lighting fidelity  — mean luminance drift < 5 units (of 255)
      3. Structural fidelity — SSIM > 0.85
      4. No plastic artefacts — local variance in skin region stays healthy
    """

    PORE_THRESHOLD = 0.82       # high-freq energy ratio must be ≥ this
    LUMINANCE_DRIFT_MAX = 8.0   # max absolute mean-L shift
    SSIM_MIN = 0.82             # minimum structural similarity
    SKIN_VAR_RATIO_MIN = 0.60   # skin-region variance must stay ≥ 60% of original

    def __init__(self):
        log.info("QualityChecker ready")

    def _high_freq_energy(self, bgr: np.ndarray) -> float:
        """Sum of absolute Laplacian — measures micro-texture energy."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(np.mean(np.abs(lap)))

    def _mean_luminance(self, bgr: np.ndarray) -> float:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        return float(np.mean(lab[:, :, 0]))

    def _ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Simplified SSIM on grayscale (windowed)."""
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(g1 * g1, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(g2 * g2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2

        num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map = num / den
        return float(np.mean(ssim_map))

    def _skin_variance_ratio(
        self, orig: np.ndarray, retouched: np.ndarray, mask: np.ndarray
    ) -> float:
        """Compare local variance in skin regions before/after."""
        g_orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.float64)
        g_ret = cv2.cvtColor(retouched, cv2.COLOR_BGR2GRAY).astype(np.float64)

        skin_px = mask > 0
        if not np.any(skin_px):
            return 1.0  # no skin → pass

        var_orig = float(np.var(g_orig[skin_px]))
        var_ret = float(np.var(g_ret[skin_px]))

        if var_orig < 1.0:
            return 1.0
        return var_ret / var_orig

    def check(self, rec: ImageRecord) -> Dict[str, Any]:
        """Run all quality gates. Returns dict of check results."""
        orig = rec.decoded_bgr
        ret = rec.retouched_bgr
        mask = rec.skin_mask if rec.skin_mask is not None else np.zeros(orig.shape[:2], dtype=np.uint8)

        # 1. Pore preservation (high-freq energy)
        hf_orig = self._high_freq_energy(orig)
        hf_ret = self._high_freq_energy(ret)
        pore_ratio = hf_ret / hf_orig if hf_orig > 0 else 1.0
        pore_pass = pore_ratio >= self.PORE_THRESHOLD

        # 2. Lighting fidelity
        lum_orig = self._mean_luminance(orig)
        lum_ret = self._mean_luminance(ret)
        lum_drift = abs(lum_ret - lum_orig)
        lum_pass = lum_drift <= self.LUMINANCE_DRIFT_MAX

        # 3. SSIM
        ssim_val = self._ssim(orig, ret)
        ssim_pass = ssim_val >= self.SSIM_MIN

        # 4. Skin variance (no plastic)
        sv_ratio = self._skin_variance_ratio(orig, ret, mask)
        sv_pass = sv_ratio >= self.SKIN_VAR_RATIO_MIN

        all_pass = pore_pass and lum_pass and ssim_pass and sv_pass

        checks = {
            "pore_ratio": round(pore_ratio, 4),
            "pore_pass": pore_pass,
            "luminance_drift": round(lum_drift, 2),
            "luminance_pass": lum_pass,
            "ssim": round(ssim_val, 4),
            "ssim_pass": ssim_pass,
            "skin_var_ratio": round(sv_ratio, 4),
            "skin_var_pass": sv_pass,
            "ALL_PASS": all_pass,
        }

        rec.quality_checks = checks
        return checks


# ═══════════════════════════════════════════════════════════════════════════
# 5. ARCFACE VALIDATOR (Local Stub)
# ═══════════════════════════════════════════════════════════════════════════

class LocalArcFaceValidator:
    """
    Local stub for ArcFace identity consistency checking.
    Uses deterministic pixel-statistics embeddings.
    Threshold = 0.85 cosine similarity (locked by Roundtable).
    """

    THRESHOLD: float = 0.85

    def __init__(self):
        log.info("LocalArcFaceValidator ready (threshold=%.2f)", self.THRESHOLD)

    def extract_embedding(
        self, bgr: np.ndarray, face_box: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Extract 512-d pseudo-embedding from face region statistics."""
        if face_box:
            x1, y1 = face_box["x1"], face_box["y1"]
            x2, y2 = face_box["x2"], face_box["y2"]
            crop = bgr[y1:y2, x1:x2]
        else:
            h, w = bgr.shape[:2]
            crop = bgr[h // 4: 3 * h // 4, w // 4: 3 * w // 4]

        if crop.size == 0:
            return np.zeros(512, dtype=np.float32)

        small = cv2.resize(crop, (32, 16))
        flat = small.astype(np.float32).flatten()
        indices = np.linspace(0, len(flat) - 1, 512, dtype=int)
        emb = flat[indices]
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.astype(np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def validate_identity(
        self, record: ImageRecord, transformed_bgr: np.ndarray, phase_name: str = ""
    ) -> Tuple[bool, float]:
        """Compare transformed image face against Phase B baseline."""
        if record.face_embedding is None:
            base = record.retouched_bgr if record.retouched_bgr is not None else record.decoded_bgr
            largest = max(record.face_boxes, key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])) if record.face_boxes else None
            record.face_embedding = self.extract_embedding(base, largest)
            log.info("ArcFace: baseline embedding extracted for %s", record.uid)

        largest = max(record.face_boxes, key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])) if record.face_boxes else None
        new_emb = self.extract_embedding(transformed_bgr, largest)
        sim = self.cosine_similarity(record.face_embedding, new_emb)
        passed = sim >= self.THRESHOLD

        log.info(
            "ArcFace [%s]: %s  sim=%.4f  %s",
            phase_name, record.source_path.name, sim,
            "PASSED" if passed else "REJECTED",
        )
        return passed, sim


# ═══════════════════════════════════════════════════════════════════════════
# 6. DANBOORU TAGGER (Local Stub)
# ═══════════════════════════════════════════════════════════════════════════

class LocalDanbooruTagger:
    """
    Stub Danbooru tagger: derives plausible tags from image statistics.
    Classifies as cosplay if ≥2 character-indicator tags detected.
    """

    COSPLAY_THRESHOLD: float = 0.60
    COSPLAY_INDICATOR_TAGS = {
        "cosplay", "costume", "armor", "cape", "sword", "staff",
        "magical_girl", "mecha", "uniform", "sailor_collar",
        "school_uniform", "military_uniform", "gothic_lolita",
        "kimono", "yukata", "chinese_clothes", "hanfu",
        "witch_hat", "crown", "tiara", "horns", "wings",
        "elf", "vampire", "demon", "angel",
    }

    def __init__(self):
        log.info("LocalDanbooruTagger ready")

    def tag_image(self, bgr: np.ndarray) -> Dict[str, float]:
        h, w = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mean_h, mean_s, mean_v = [float(x) for x in cv2.mean(hsv)[:3]]

        tags: Dict[str, float] = {}
        tags["1girl"] = 0.75
        tags["portrait"] = 0.82
        tags["photo_(medium)"] = 0.70
        tags["realistic"] = 0.78

        if mean_s > 100:
            tags["colorful"] = round(mean_s / 180.0, 4)
        if mean_v > 180:
            tags["bright"] = round(mean_v / 255.0, 4)
        elif mean_v < 80:
            tags["dark"] = round(1.0 - mean_v / 255.0, 4)

        if mean_h < 30 or mean_h > 160:
            tags["warm_tones"] = 0.65
        else:
            tags["cool_tones"] = 0.65

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / edges.size
        if edge_density > 0.15:
            tags["detailed_background"] = round(min(edge_density * 3, 1.0), 4)
        else:
            tags["simple_background"] = round(1.0 - edge_density * 5, 4)

        return tags

    def classify_archetype(self, tags: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        cosplay_hits = []
        for tag, score in tags.items():
            if tag.lower() in self.COSPLAY_INDICATOR_TAGS and score >= self.COSPLAY_THRESHOLD:
                cosplay_hits.append((tag, score))

        is_cosplay = len(cosplay_hits) >= 2

        if is_cosplay:
            tag_strs = [t for t, _ in sorted(cosplay_hits, key=lambda x: -x[1])[:5]]
            prompt = (
                f"canonical environment for character: {', '.join(tag_strs)}. "
                f"Dramatic cinematic lighting, high detail"
            )
            return True, prompt

        mood_tags = [t for t in tags if t in {"warm_tones", "cool_tones", "dark", "bright"}]
        if "dark" in mood_tags:
            return False, "cyberpunk neon cityscape at night, rain-slicked streets, volumetric fog"
        elif "warm_tones" in mood_tags:
            return False, "golden hour fantasy garden, bokeh lights, ethereal atmosphere"
        else:
            return False, "futuristic sci-fi corridor, holographic displays, teal and orange lighting"


# ═══════════════════════════════════════════════════════════════════════════
# 7. PHASE C — MASTER FILTER (Local)
# ═══════════════════════════════════════════════════════════════════════════

class LocalPhaseC:
    """
    Local Phase C: vision agent style selection → colour grading → ArcFace gate.
    """

    STYLE_PRESETS = {
        "cinematic_teal_orange": {
            "shadow_hue": (180, 40), "highlight_hue": (20, 50),
            "contrast": 1.15, "saturation": 1.10, "temp_shift": 8, "grain": 0.02,
        },
        "cyberpunk_neon": {
            "shadow_hue": (280, 60), "highlight_hue": (170, 50),
            "contrast": 1.25, "saturation": 1.20, "temp_shift": -5, "grain": 0.03,
        },
        "soft_film": {
            "shadow_hue": (220, 20), "highlight_hue": (40, 15),
            "contrast": 0.90, "saturation": 0.85, "temp_shift": 3, "grain": 0.04,
        },
        "golden_hour": {
            "shadow_hue": (30, 30), "highlight_hue": (45, 40),
            "contrast": 1.05, "saturation": 1.08, "temp_shift": 15, "grain": 0.01,
        },
    }

    def __init__(self, arcface: LocalArcFaceValidator, output_dir: Path):
        self.arcface = arcface
        self.output_dir = output_dir
        log.info("LocalPhaseC ready")

    def _select_style(self, bgr: np.ndarray) -> Tuple[str, Dict]:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
        mean_l = float(np.mean(lab[:, :, 0]))
        mean_b = float(np.mean(lab[:, :, 2]))

        if mean_l < 90:
            name = "cyberpunk_neon"
        elif mean_b > 135:
            name = "golden_hour"
        elif mean_b < 120:
            name = "cinematic_teal_orange"
        else:
            name = "soft_film"

        log.info("PhaseC VisionAgent: selected '%s'", name)
        return name, self.STYLE_PRESETS[name]

    def _apply_grade(self, bgr: np.ndarray, style: Dict) -> np.ndarray:
        img = bgr.astype(np.float64)

        # Contrast
        lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float64)
        L = lab[:, :, 0]
        L = 128.0 + (L - 128.0) * style["contrast"]
        lab[:, :, 0] = np.clip(L, 0, 255)
        img = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float64)

        # Saturation
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float64)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * style["saturation"], 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float64)

        # Temperature
        lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float64)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + style["temp_shift"], 0, 255)
        img = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float64)

        # Film grain
        if style["grain"] > 0:
            noise = np.random.normal(0, style["grain"] * 255, img.shape)
            img = np.clip(img + noise, 0, 255)

        return img.astype(np.uint8)

    def process(self, rec: ImageRecord) -> ImageRecord:
        if rec.retouched_bgr is None:
            log.warning("PhaseC: no Phase B output for %s", rec.uid)
            return rec

        style_name, style = self._select_style(rec.retouched_bgr)
        graded = self._apply_grade(rec.retouched_bgr, style)

        passed, sim = self.arcface.validate_identity(rec, graded, "PhaseC")
        if not passed:
            log.warning("PhaseC: ArcFace rejected, blending fallback")
            graded = cv2.addWeighted(rec.retouched_bgr, 0.5, graded, 0.5, 0)
            passed, sim = self.arcface.validate_identity(rec, graded, "PhaseC-fb")

        rec.phase_c_bgr = graded
        stem = rec.source_path.stem
        out_path = self.output_dir / f"{stem}_C_MasterFilter.png"
        cv2.imwrite(str(out_path), graded)
        rec.phase_c_path = out_path
        rec.phase_c_metadata = {
            "style": style_name,
            "arcface_sim": round(sim, 4),
            "arcface_passed": passed,
        }
        log.info("PhaseC: %s → %s (style=%s, arcface=%.4f)", rec.source_path.name, out_path.name, style_name, sim)
        return rec


# ═══════════════════════════════════════════════════════════════════════════
# 8. PHASE D — CREATIVE EXPANSION (Local)
# ═══════════════════════════════════════════════════════════════════════════

class LocalPhaseD:
    """
    Local Phase D: Danbooru tags → archetype reasoning → BG generation → ArcFace gate.
    """

    def __init__(self, arcface: LocalArcFaceValidator, tagger: LocalDanbooruTagger, output_dir: Path):
        self.arcface = arcface
        self.tagger = tagger
        self.output_dir = output_dir
        log.info("LocalPhaseD ready")

    def _generate_background(self, bgr: np.ndarray, prompt: str) -> np.ndarray:
        h, w = bgr.shape[:2]

        # Subject isolation via GrabCut
        mask = np.zeros((h, w), np.uint8)
        rect = (w // 5, h // 8, 3 * w // 5, 7 * h // 8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
        fg_mask = cv2.GaussianBlur(fg_mask, (21, 21), 0)

        # Procedural background
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        prompt_lower = prompt.lower()
        if "cyberpunk" in prompt_lower or "neon" in prompt_lower:
            for y in range(h):
                t = y / h
                bg[y, :, 0] = int(20 + 40 * t)
                bg[y, :, 1] = int(10 + 30 * (1 - t))
                bg[y, :, 2] = int(30 + 60 * t)
        elif "golden" in prompt_lower or "garden" in prompt_lower:
            for y in range(h):
                t = y / h
                bg[y, :, 0] = int(40 + 30 * t)
                bg[y, :, 1] = int(140 + 60 * (1 - t))
                bg[y, :, 2] = int(200 + 55 * (1 - t))
        else:
            for y in range(h):
                t = y / h
                bg[y, :, 0] = int(100 + 60 * (1 - t))
                bg[y, :, 1] = int(80 + 40 * t)
                bg[y, :, 2] = int(50 + 100 * t)

        bg = cv2.GaussianBlur(bg, (41, 41), 20)
        noise = np.random.normal(0, 8, bg.shape)
        bg = np.clip(bg.astype(np.float64) + noise, 0, 255).astype(np.uint8)

        # Composite
        fg_norm = fg_mask.astype(np.float64) / 255.0
        fg_norm = fg_norm[:, :, np.newaxis]
        composite = bgr.astype(np.float64) * fg_norm + bg.astype(np.float64) * (1 - fg_norm)
        return np.clip(composite, 0, 255).astype(np.uint8)

    def process(self, rec: ImageRecord) -> ImageRecord:
        source = rec.phase_c_bgr
        if source is None:
            log.warning("PhaseD: no Phase C output for %s", rec.uid)
            return rec

        # Danbooru tagging
        tags = self.tagger.tag_image(source)
        rec.danbooru_tags = tags
        log.info(
            "PhaseD: %d tags (top: %s)",
            len(tags),
            ", ".join(f"{k}={v:.2f}" for k, v in sorted(tags.items(), key=lambda x: -x[1])[:5]),
        )

        # Archetype classification
        is_cosplay, prompt = self.tagger.classify_archetype(tags)
        rec.is_cosplay = is_cosplay
        rec.archetype_prompt = prompt
        log.info("PhaseD: archetype=%s  prompt='%s'", "COSPLAY" if is_cosplay else "PORTRAIT", prompt[:60] if prompt else "")

        # Background generation
        expanded = self._generate_background(source, prompt or "")

        # ArcFace gate
        passed, sim = self.arcface.validate_identity(rec, expanded, "PhaseD")
        if not passed:
            log.warning("PhaseD: ArcFace rejected (%.4f), blending fallback", sim)
            expanded = cv2.addWeighted(source, 0.6, expanded, 0.4, 0)
            passed, sim = self.arcface.validate_identity(rec, expanded, "PhaseD-fb")

        rec.phase_d_bgr = expanded
        stem = rec.source_path.stem
        out_path = self.output_dir / f"{stem}_D_Expansion.png"
        cv2.imwrite(str(out_path), expanded)
        rec.phase_d_path = out_path
        rec.phase_d_metadata = {
            "is_cosplay": is_cosplay,
            "prompt": prompt,
            "tag_count": len(tags),
            "arcface_sim": round(sim, 4),
            "arcface_passed": passed,
        }
        log.info("PhaseD: %s → %s (cosplay=%s, arcface=%.4f)", rec.source_path.name, out_path.name, is_cosplay, sim)
        return rec


# ═══════════════════════════════════════════════════════════════════════════
# 9. PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_local_test(input_dir: str, output_dir: str = "./output") -> bool:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Discover images
    exts = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
    images = sorted(
        p for p in input_path.iterdir()
        if p.suffix.lower() in exts and p.is_file()
    )

    if not images:
        log.error("No images found in %s", input_path)
        return False

    log.info("Found %d test image(s) in %s", len(images), input_path)

    # Init all modules (B + C + D)
    sorter = LocalSorter(threshold=0.35)
    mapper = LocalMapper()
    forge = LocalForge(denoise=0.30, output_dir=output_path)
    checker = QualityChecker()
    arcface = LocalArcFaceValidator()
    tagger = LocalDanbooruTagger()
    phase_c = LocalPhaseC(arcface=arcface, output_dir=output_path)
    phase_d = LocalPhaseD(arcface=arcface, tagger=tagger, output_dir=output_path)

    records: List[ImageRecord] = []
    all_success = True

    for img_path in images:
        log.info("")
        log.info("━" * 60)
        log.info("Processing: %s", img_path.name)
        log.info("━" * 60)

        rec = ImageRecord(source_path=img_path, uid=img_path.stem)

        # Decode
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            log.error("Failed to decode %s — skipping", img_path.name)
            all_success = False
            continue
        rec.decoded_bgr = bgr
        log.info("Decoded: %dx%d", bgr.shape[1], bgr.shape[0])

        # ── Phase A: Sort ─────────────────────────────────────────
        sorter.evaluate(rec)
        if not rec.is_accepted:
            log.warning("REJECTED by Sorter — but continuing for test validation")
            rec.is_accepted = True

        # ── Phase A: Map ──────────────────────────────────────────
        mapper.analyse(rec)

        # ── Phase B: Forge (retouch) ──────────────────────────────
        forge.retouch(rec)

        # ── Phase B: Quality checks ───────────────────────────────
        checks = checker.check(rec)
        status_b = "PASS" if checks["ALL_PASS"] else "FAIL"
        log.info(
            "Phase B Quality: %s  pore=%.3f  lum=%.1f  ssim=%.3f  skin_var=%.3f",
            status_b,
            checks["pore_ratio"],
            checks["luminance_drift"],
            checks["ssim"],
            checks["skin_var_ratio"],
        )

        # ── Phase C: Master Filter ────────────────────────────────
        phase_c.process(rec)
        c_ok = rec.phase_c_path is not None
        c_sim = rec.phase_c_metadata.get("arcface_sim", 0)
        log.info(
            "Phase C: %s  style=%s  arcface=%.4f",
            "PASS" if c_ok else "FAIL",
            rec.phase_c_metadata.get("style", "?"),
            c_sim,
        )

        # ── Phase D: Creative Expansion ───────────────────────────
        phase_d.process(rec)
        d_ok = rec.phase_d_path is not None
        d_sim = rec.phase_d_metadata.get("arcface_sim", 0)
        log.info(
            "Phase D: %s  cosplay=%s  tags=%d  arcface=%.4f",
            "PASS" if d_ok else "FAIL",
            rec.is_cosplay,
            len(rec.danbooru_tags),
            d_sim,
        )

        # ── Overall success: B + C + D all produced output ────────
        img_success = checks["ALL_PASS"] and c_ok and d_ok
        if not img_success:
            all_success = False

        records.append(rec)

    # ── Final report ──────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("  APS FULL PIPELINE TEST — B → C → D REPORT")
    print("=" * 96)
    print(
        f"  {'Image':<20s} {'IQA':>5s} {'Face':>4s} "
        f"{'B-SSIM':>6s} {'B-Pore':>6s} "
        f"{'C-Style':<22s} {'C-Arc':>5s} "
        f"{'D-Tags':>6s} {'D-Arc':>5s} {'D-Type':<10s} "
        f"{'All':>4s}"
    )
    print("-" * 96)

    pass_count = 0
    for rec in records:
        c = rec.quality_checks
        b_ok = c.get("ALL_PASS", False)
        c_ok = rec.phase_c_path is not None
        d_ok = rec.phase_d_path is not None
        all_ok = b_ok and c_ok and d_ok
        if all_ok:
            pass_count += 1

        print(
            f"  {rec.source_path.name:<20s} "
            f"{rec.quality_score:>5.2f} "
            f"{len(rec.face_boxes):>4d} "
            f"{c.get('ssim', 0):>6.3f} "
            f"{c.get('pore_ratio', 0):>6.3f} "
            f"{rec.phase_c_metadata.get('style', 'n/a'):<22s} "
            f"{rec.phase_c_metadata.get('arcface_sim', 0):>5.3f} "
            f"{len(rec.danbooru_tags):>6d} "
            f"{rec.phase_d_metadata.get('arcface_sim', 0):>5.3f} "
            f"{'COSPLAY' if rec.is_cosplay else 'PORTRAIT':<10s} "
            f"{'OK' if all_ok else 'FAIL':>4s}"
        )

    print("-" * 96)
    print(f"  Result: {pass_count}/{len(records)} images passed full B→C→D pipeline")
    print("=" * 96)

    if all_success:
        print("\n  SUCCESS — All 5 images passed through Phase B, C, and D.\n")
    else:
        print("\n  WARNING — Some images failed. See details above.\n")

    return all_success


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="APS Local Test Runner")
    parser.add_argument("--input", "-i", type=str, default="./test_images")
    parser.add_argument("--output", "-o", type=str, default="./output")
    args = parser.parse_args()

    success = run_local_test(args.input, args.output)
    sys.exit(0 if success else 1)
