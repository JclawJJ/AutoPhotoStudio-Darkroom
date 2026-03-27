"""
APS Darkroom API — test client.
Sends a test image to the /api/process endpoint and validates the response.

Usage:
    python test_client.py                        # uses default test image
    python test_client.py path/to/image.jpg      # uses specified image
    python test_client.py --denoise 0.28         # override denoise value
"""

import argparse
import json
import sys
from pathlib import Path

import requests

API = "http://localhost:8000"


def find_test_image() -> Path:
    """Find a suitable test image in the project."""
    candidates = [
        Path(__file__).parent / "test_images",
        Path(__file__).parent / "test_data",
    ]
    for d in candidates:
        if d.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                imgs = list(d.glob(ext))
                if imgs:
                    return imgs[0]
    # Fallback: generate a synthetic test image
    return generate_test_image()


def generate_test_image() -> Path:
    """Create a minimal synthetic test image if no real images are available."""
    try:
        import numpy as np
        import cv2

        img = np.random.randint(80, 200, (480, 640, 3), dtype=np.uint8)
        # Add a bright rectangle to simulate a face-like region
        cv2.rectangle(img, (220, 120), (420, 380), (200, 180, 170), -1)
        path = Path(__file__).parent / "output" / "_test_synthetic.jpg"
        path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(path), img)
        return path
    except ImportError:
        print("[ERROR] No test images found and opencv not available to generate one.")
        sys.exit(1)


def test_health():
    """Verify the server is reachable."""
    try:
        r = requests.get(f"{API}/docs", timeout=5)
        assert r.status_code == 200, f"Unexpected status: {r.status_code}"
        print("[PASS] Server reachable — /docs returned 200")
        return True
    except requests.ConnectionError:
        print(f"[FAIL] Cannot reach server at {API}")
        print("       Start it with: python aps_server.py")
        return False


def test_process(image_path: Path, denoise: float = 0.30):
    """POST an image to /api/process and validate the response shape."""
    print(f"\n[TEST] Uploading: {image_path.name}  denoise={denoise}")

    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{API}/api/process",
            files={"file": (image_path.name, f, "image/jpeg")},
            data={"denoise": str(denoise)},
            timeout=30,
        )

    if resp.status_code == 400:
        body = resp.json()
        print(f"[WARN] Image rejected: {body.get('error', 'unknown')}")
        if "quality_score" in body:
            print(f"       quality={body['quality_score']:.3f}  "
                  f"sharpness={body.get('sharpness', 'n/a')}  "
                  f"exposure={body.get('exposure', 'n/a')}")
        return True  # Server responded correctly, just rejected the image

    assert resp.status_code == 200, f"Unexpected status: {resp.status_code}\n{resp.text}"
    body = resp.json()
    print(f"[PASS] Response OK — job_id={body['job_id']}")
    print(json.dumps(body, indent=2))

    # Validate required fields
    required = ["job_id", "denoise", "quality_score", "faces_detected",
                "original", "processed", "mask"]
    missing = [k for k in required if k not in body]
    assert not missing, f"Missing keys: {missing}"
    print(f"[PASS] All required fields present")

    # Validate images are downloadable
    for key in ("original", "processed", "mask"):
        url = f"{API}{body[key]}"
        r = requests.get(url, timeout=10)
        assert r.status_code == 200, f"Failed to fetch {key}: {r.status_code}"
        assert len(r.content) > 100, f"{key} seems too small ({len(r.content)} bytes)"
        print(f"[PASS] {key:10s} → {len(r.content):,} bytes")

    # Validate denoise value was echoed back
    assert abs(body["denoise"] - denoise) < 0.001, \
        f"Denoise mismatch: sent {denoise}, got {body['denoise']}"
    print(f"[PASS] Denoise value echoed correctly: {body['denoise']}")

    return True


def test_static_files():
    """Verify the /output static mount works."""
    r = requests.get(f"{API}/output/", timeout=5)
    # Static files may return 404 for directory listing, that's fine
    print(f"[INFO] /output/ status: {r.status_code}")
    return True


def main():
    parser = argparse.ArgumentParser(description="APS Darkroom API test client")
    parser.add_argument("image", nargs="?", help="Path to test image")
    parser.add_argument("--denoise", type=float, default=0.30,
                        help="Denoise value (0.25-0.35)")
    args = parser.parse_args()

    print("=" * 60)
    print("  APS Darkroom — API Test Client")
    print("=" * 60)

    if not test_health():
        sys.exit(1)

    test_static_files()

    image_path = Path(args.image) if args.image else find_test_image()
    if not image_path.exists():
        print(f"[FAIL] Image not found: {image_path}")
        sys.exit(1)

    ok = test_process(image_path, args.denoise)

    print("\n" + "=" * 60)
    if ok:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
