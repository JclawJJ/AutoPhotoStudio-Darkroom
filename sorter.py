#!/usr/bin/env python3
"""
sorter.py — Phase 2: The Blurs Sorter (Laplacian Variance)
===========================================================
Reads images from test_data/raws/.
Calculates Laplacian variance to detect blur.
Blurry images (variance < threshold) are moved to test_data/rejected/.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(name)s]  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("APS.Sorter")

# Default blur threshold — images with Laplacian variance below this are rejected.
# Typical sharp portrait: 200-2000+. Blurry: < 100.
DEFAULT_BLUR_THRESHOLD = 100.0


class Sorter:
    """
    OpenCV Laplacian Variance blur detector.

    Computes var(Laplacian(gray)) for each image.
    Low variance = blurry = rejected.
    """

    def __init__(self, threshold: float = DEFAULT_BLUR_THRESHOLD):
        self.threshold = threshold
        log.info("Sorter ready: Laplacian variance threshold=%.1f", self.threshold)

    def compute_blur_score(self, image_path: str) -> float:
        """
        Compute the Laplacian variance blur score for an image.

        Higher values = sharper image. Lower values = blurrier.
        """
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(laplacian.var())
        return variance

    def is_sharp(self, image_path: str) -> Tuple[bool, float]:
        """
        Check if an image passes the sharpness gate.

        Returns:
            (accepted, variance) — True if sharp enough, plus the raw score.
        """
        variance = self.compute_blur_score(image_path)
        accepted = variance >= self.threshold
        stem = Path(image_path).name
        status = "ACCEPTED" if accepted else "REJECTED (blurry)"
        log.info("Sorter: %-25s  var=%.2f  %s", stem, variance, status)
        return accepted, variance

    def sort_directory(
        self,
        input_dir: str = "test_data/raws",
        rejected_dir: str = "test_data/rejected",
    ) -> Dict[str, List[str]]:
        """
        Scan all images in input_dir. Move blurry ones to rejected_dir.

        Returns:
            {"accepted": [...paths], "rejected": [...paths]}
        """
        src = Path(input_dir)
        rej = Path(rejected_dir)
        rej.mkdir(parents=True, exist_ok=True)

        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        images = sorted(
            p for p in src.iterdir()
            if p.is_file() and p.suffix.lower() in valid_ext
        )

        if not images:
            log.warning("No images found in %s", src)
            return {"accepted": [], "rejected": []}

        log.info("Sorter: scanning %d image(s) in %s", len(images), src)

        accepted: List[str] = []
        rejected: List[str] = []

        for img_path in images:
            try:
                sharp, variance = self.is_sharp(str(img_path))
                if sharp:
                    accepted.append(str(img_path))
                else:
                    dest = rej / img_path.name
                    shutil.move(str(img_path), str(dest))
                    rejected.append(str(dest))
                    log.info("Sorter: moved %s → %s", img_path.name, rej)
            except Exception as e:
                log.error("Sorter: failed on %s: %s", img_path.name, e)

        log.info(
            "Sorter summary: %d accepted, %d rejected out of %d",
            len(accepted), len(rejected), len(images),
        )
        return {"accepted": accepted, "rejected": rejected}


if __name__ == "__main__":
    import sys

    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BLUR_THRESHOLD
    sorter = Sorter(threshold=threshold)
    results = sorter.sort_directory()

    print(f"\n{'='*60}")
    print(f"  Sorter Results (threshold={threshold})")
    print(f"{'='*60}")
    print(f"  Accepted: {len(results['accepted'])}")
    for p in results["accepted"]:
        print(f"    ✓ {p}")
    print(f"  Rejected: {len(results['rejected'])}")
    for p in results["rejected"]:
        print(f"    ✗ {p}")
    print(f"{'='*60}")
