#!/usr/bin/env python3
"""
test_aps.py — Phase 4: Full Local Audit / Dry-Run Test
=======================================================
Exercises all three APS backend phases end-to-end:
  Phase 1: Mapper  (MediaPipe segmentation → masks)
  Phase 2: Sorter  (Laplacian variance blur detection)
  Phase 3: Forge   (ComfyUI workflow generation + fire attempt)

Uses test_data/raws/ as input. Creates a fake black JPG if none exists.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

# ── Ensure we're running from the project root ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

# ── Directories ──────────────────────────────────────────────────────────
RAW_DIR = PROJECT_ROOT / "test_data" / "raws"
MASK_DIR = PROJECT_ROOT / "test_data" / "masks"
REJECTED_DIR = PROJECT_ROOT / "test_data" / "rejected"
FAKE_IMAGE = RAW_DIR / "fake_black.jpg"


def ensure_test_data():
    """Create directories and a fake black JPG if needed."""
    for d in [RAW_DIR, MASK_DIR, REJECTED_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    if not any(RAW_DIR.glob("*.jpg")) and not any(RAW_DIR.glob("*.png")):
        print("[SETUP] Creating fake_black.jpg (640x480 black) for testing…")
        from PIL import Image
        img = Image.new("RGB", (640, 480), color=(0, 0, 0))
        img.save(str(FAKE_IMAGE))
        print(f"[SETUP] Created: {FAKE_IMAGE}")


def test_phase1_mapper():
    """Phase 1: MediaPipe Mapper — extract skin + background masks."""
    print("\n" + "=" * 70)
    print("  PHASE 1: THE MAPPER (MediaPipe Segmentation)")
    print("=" * 70)

    from mapper import Mapper

    mapper = Mapper(model_cache_dir=str(PROJECT_ROOT / ".aps_cache" / "models"))

    images = sorted(RAW_DIR.glob("*"))
    images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    if not images:
        print("[WARN] No images in test_data/raws/ — skipping mapper test")
        return False

    results = {}
    for img_path in images:
        print(f"\n  Processing: {img_path.name}")
        skin_mask, bg_mask = mapper.extract_masks(str(img_path), str(MASK_DIR))
        results[img_path.name] = {
            "skin_pixels": int((skin_mask > 0).sum()),
            "bg_pixels": int((bg_mask > 0).sum()),
            "total_pixels": skin_mask.size,
        }

    print("\n  Mapper Results:")
    for name, r in results.items():
        skin_pct = r["skin_pixels"] / r["total_pixels"] * 100
        bg_pct = r["bg_pixels"] / r["total_pixels"] * 100
        print(f"    {name}: skin={skin_pct:.1f}%, bg={bg_pct:.1f}%")

    # Verify mask files were saved
    mask_files = list(MASK_DIR.glob("*.png"))
    print(f"\n  Mask files saved: {len(mask_files)}")
    for mf in mask_files:
        print(f"    → {mf.name}")

    print("\n  [PHASE 1] PASS ✓")
    return True


def test_phase2_sorter():
    """Phase 2: The Sorter — Laplacian variance blur detection."""
    print("\n" + "=" * 70)
    print("  PHASE 2: THE SORTER (Laplacian Variance Blur Filter)")
    print("=" * 70)

    from sorter import Sorter

    # Use a very low threshold so our fake black image isn't auto-rejected
    # (black image has variance=0, so it WILL be rejected at default threshold)
    sorter = Sorter(threshold=0.0)

    images = sorted(RAW_DIR.glob("*"))
    images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    if not images:
        print("[WARN] No images in test_data/raws/ — skipping sorter test")
        return False

    for img_path in images:
        accepted, variance = sorter.is_sharp(str(img_path))
        status = "SHARP" if accepted else "BLURRY"
        print(f"    {img_path.name}: variance={variance:.2f} → {status}")

    # Now test the directory sort with a realistic threshold
    print("\n  Testing directory sort (threshold=100)…")
    strict_sorter = Sorter(threshold=100.0)
    # Don't actually move files in test — just compute scores
    for img_path in images:
        score = strict_sorter.compute_blur_score(str(img_path))
        verdict = "ACCEPT" if score >= 100.0 else "REJECT"
        print(f"    {img_path.name}: score={score:.2f} → {verdict}")

    print("\n  [PHASE 2] PASS ✓")
    return True


def test_phase3_forge():
    """Phase 3: ComfyUI Forge — workflow generation + fire attempt."""
    print("\n" + "=" * 70)
    print("  PHASE 3: THE FORGE (ComfyUI Workflow Generator)")
    print("=" * 70)

    from forge import Forge

    forge = Forge()

    # Use the fake image and a mask (or create one)
    test_img = str(FAKE_IMAGE)
    mask_candidates = list(MASK_DIR.glob("*_skin_mask.png"))
    if mask_candidates:
        test_mask = str(mask_candidates[0])
    else:
        # Create a dummy mask
        import numpy as np
        import cv2
        dummy_mask = np.zeros((480, 640), dtype=np.uint8)
        dummy_mask_path = MASK_DIR / "dummy_skin_mask.png"
        cv2.imwrite(str(dummy_mask_path), dummy_mask)
        test_mask = str(dummy_mask_path)

    # 1. Build workflow
    print("\n  Building workflow…")
    payload = forge.build_workflow(test_img, test_mask)
    nodes = payload["prompt"]
    print(f"    Nodes: {len(nodes)}")
    print(f"    Client ID: {payload['client_id']}")
    print(f"    Denoise: {forge.denoise}")
    print(f"    Steps: {forge.steps}")

    # Verify denoise lock
    ksampler_denoise = nodes["8"]["inputs"]["denoise"]
    facedetailer_denoise = nodes["9"]["inputs"]["denoise"]
    assert 0.25 <= ksampler_denoise <= 0.35, f"KSampler denoise out of range: {ksampler_denoise}"
    assert 0.25 <= facedetailer_denoise <= 0.35, f"FaceDetailer denoise out of range: {facedetailer_denoise}"
    print(f"    Denoise lock verified: KSampler={ksampler_denoise}, FaceDetailer={facedetailer_denoise}")

    # Verify key nodes exist
    required_nodes = [
        ("1", "CheckpointLoaderSimple"),
        ("2", "CLIPTextEncode"),
        ("3", "CLIPTextEncode"),
        ("4", "WAS_Image_Load"),
        ("5", "WAS_Image_Load"),
        ("6", "ImageToMask"),
        ("7", "VAEEncode"),
        ("8", "KSampler"),
        ("9", "FaceDetailer"),
        ("10", "SetLatentNoiseMask"),
        ("11", "VAEDecode"),
        ("12", "ImageBlend"),
        ("13", "WAS_Image_Save"),
        ("15", "UltralyticsDetectorProvider"),
    ]
    for node_id, class_type in required_nodes:
        assert node_id in nodes, f"Missing node {node_id}"
        assert nodes[node_id]["class_type"] == class_type, (
            f"Node {node_id} expected {class_type}, got {nodes[node_id]['class_type']}"
        )
    print(f"    All {len(required_nodes)} required nodes verified ✓")

    # 2. Save workflow JSON
    print("\n  Saving workflow JSON…")
    out_json = str(PROJECT_ROOT / "test_data" / "workflow_debug.json")
    forge.save_workflow_json(test_img, test_mask, out_json)
    assert Path(out_json).exists(), "Workflow JSON not saved"
    print(f"    Saved: {out_json}")

    # 3. Fire at ComfyUI (expect offline)
    print("\n  Firing at ComfyUI (expecting offline)…")
    result = forge.fire(test_img, test_mask)
    print(f"    Status: {result['status']}")
    if result["status"] == "offline":
        print("    ComfyUI offline — expected. Workflow logic preserved.")
    elif result["status"] == "queued":
        print(f"    Queued! prompt_id={result['prompt_id']}")

    print("\n  [PHASE 3] PASS ✓")
    return True


def test_full_pipeline_hook():
    """Integration: Mapper → Sorter → Forge chained together."""
    print("\n" + "=" * 70)
    print("  INTEGRATION: MAPPER → SORTER → FORGE")
    print("=" * 70)

    from mapper import Mapper
    from sorter import Sorter
    from forge import Forge

    mapper = Mapper(model_cache_dir=str(PROJECT_ROOT / ".aps_cache" / "models"))
    sorter = Sorter(threshold=0.0)  # permissive for black test image
    forge = Forge()

    images = sorted(RAW_DIR.glob("*"))
    images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    for img_path in images:
        name = img_path.name
        print(f"\n  ── {name} ──")

        # Phase 2: Sort
        accepted, variance = sorter.is_sharp(str(img_path))
        print(f"    Sorter: var={variance:.2f} → {'ACCEPT' if accepted else 'REJECT'}")
        if not accepted:
            print(f"    Skipping (blurry)")
            continue

        # Phase 1: Map
        skin_mask, bg_mask = mapper.extract_masks(str(img_path), str(MASK_DIR))
        skin_pct = (skin_mask > 0).sum() / skin_mask.size * 100
        print(f"    Mapper: skin={skin_pct:.1f}%")

        # Phase 3: Forge
        mask_path = MASK_DIR / f"{img_path.stem}_skin_mask.png"
        result = forge.fire(str(img_path), str(mask_path))
        print(f"    Forge:  status={result['status']}")

    print("\n  [INTEGRATION] PASS ✓")
    return True


def main():
    print("=" * 70)
    print("  APS BACKEND DRY-RUN TEST")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Python: {sys.version}")
    print("=" * 70)

    ensure_test_data()

    results = {}
    t0 = time.time()

    for name, test_fn in [
        ("Phase 1: Mapper", test_phase1_mapper),
        ("Phase 2: Sorter", test_phase2_sorter),
        ("Phase 3: Forge", test_phase3_forge),
        ("Integration", test_full_pipeline_hook),
    ]:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}")
            traceback.print_exc()
            results[name] = False

    elapsed = time.time() - t0

    # ── Final Report ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL REPORT")
    print("=" * 70)
    all_pass = True
    for name, passed in results.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        if not passed:
            all_pass = False
        print(f"    {name:<30s} {status}")
    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  Overall: {'ALL PASS ✓' if all_pass else 'SOME FAILURES ✗'}")
    print("=" * 70)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
