#!/usr/bin/env python3
"""
forge.py — Phase 3: ComfyUI Headless Forge Workflow Generator
==============================================================
Assembles a ComfyUI JSON workflow payload for portrait retouching:
  - Denoise locked aggressively to 0.25-0.35 (prevents AI plastic skin)
  - Source image + skin mask → masked inpainting + FaceDetailer
  - Fires headlessly to http://127.0.0.1:8188 (ComfyUI default)

If ComfyUI is offline, logs connection refused but maintains full logic.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(name)s]  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("APS.Forge")

# ── Aesthetic defaults ────────────────────────────────────────────────────
DENOISE_MIN = 0.25
DENOISE_MAX = 0.35
DEFAULT_DENOISE = 0.30
DEFAULT_STEPS = 20
DEFAULT_CFG = 7.0
DEFAULT_SAMPLER = "euler_ancestral"
DEFAULT_SCHEDULER = "normal"
DEFAULT_GUIDE_SIZE = 384
DEFAULT_CHECKPOINT = "realisticVisionV60B1_v51VAE.safetensors"

POSITIVE_PROMPT = (
    "lightly even out skin tone and remove redness, "
    "preserve natural skin texture and facial details, "
    "asian aesthetic, macro photography, highly detailed"
)
NEGATIVE_PROMPT = (
    "glamour retouching, heavy makeup, plastic skin, "
    "silicone reflection, glossy face, low detail"
)


class Forge:
    """
    ComfyUI headless API workflow generator and executor.
    Denoise is hard-locked to [0.25, 0.35] — no exceptions.
    """

    def __init__(
        self,
        comfyui_url: str = "http://127.0.0.1:8188",
        denoise: float = DEFAULT_DENOISE,
        steps: int = DEFAULT_STEPS,
        cfg_scale: float = DEFAULT_CFG,
        sampler: str = DEFAULT_SAMPLER,
        scheduler: str = DEFAULT_SCHEDULER,
        guide_size: int = DEFAULT_GUIDE_SIZE,
        checkpoint: str = DEFAULT_CHECKPOINT,
        positive_prompt: str = POSITIVE_PROMPT,
        negative_prompt: str = NEGATIVE_PROMPT,
        strip_metadata: bool = True,
    ):
        self.api_url = comfyui_url
        # Hard-lock denoise to [0.25, 0.35]
        self.denoise = max(DENOISE_MIN, min(denoise, DENOISE_MAX))
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.sampler = sampler
        self.scheduler = scheduler
        self.guide_size = guide_size
        self.checkpoint = checkpoint
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt
        self.strip_metadata = strip_metadata

        log.info(
            "Forge ready: api=%s, denoise=%.2f, steps=%d, cfg=%.1f, sampler=%s",
            self.api_url, self.denoise, self.steps, self.cfg_scale, self.sampler,
        )

    def build_workflow(self, image_path: str, mask_path: str) -> Dict[str, Any]:
        """
        Assemble the complete ComfyUI workflow JSON.

        Graph topology:
          [1] CheckpointLoader → model, clip, vae
          [2] CLIPTextEncode (positive) ← clip
          [3] CLIPTextEncode (negative) ← clip
          [4] WAS_Image_Load (source photo)
          [5] WAS_Image_Load (skin mask)
          [6] ImageToMask ← mask image
          [7] VAEEncode ← source pixels + vae
          [10] SetLatentNoiseMask ← latent + mask
          [8] KSampler (masked inpaint) ← model, pos, neg, masked latent
          [11] VAEDecode ← sampled latent
          [9] FaceDetailer ← source image, model, clip, vae, pos, neg, bbox_detector
          [12] ImageBlend ← FaceDetailer + skin-smoothed
          [13] WAS_Image_Save ← blended output
          [15] UltralyticsDetectorProvider (for FaceDetailer)
        """
        client_id = uuid.uuid4().hex[:16]
        seed = int(time.time()) % (2**32)

        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": self.checkpoint},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": self.positive_prompt, "clip": ["1", 1]},
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": self.negative_prompt, "clip": ["1", 1]},
            },
            "4": {
                "class_type": "WAS_Image_Load",
                "inputs": {"image_path": image_path},
            },
            "5": {
                "class_type": "WAS_Image_Load",
                "inputs": {"image_path": mask_path},
            },
            "6": {
                "class_type": "ImageToMask",
                "inputs": {"image": ["5", 0], "channel": "red"},
            },
            "7": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["4", 0], "vae": ["1", 2]},
            },
            "10": {
                "class_type": "SetLatentNoiseMask",
                "inputs": {"samples": ["7", 0], "mask": ["6", 0]},
            },
            "8": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["10", 0],
                    "seed": seed,
                    "steps": self.steps,
                    "cfg": self.cfg_scale,
                    "sampler_name": self.sampler,
                    "scheduler": self.scheduler,
                    "denoise": self.denoise,
                },
            },
            "11": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["8", 0], "vae": ["1", 2]},
            },
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
                    "seed": seed,
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
            "12": {
                "class_type": "ImageBlend",
                "inputs": {
                    "image1": ["9", 0],
                    "image2": ["11", 0],
                    "blend_factor": 0.65,
                    "blend_mode": "normal",
                },
            },
            "13": {
                "class_type": "WAS_Image_Save",
                "inputs": {
                    "images": ["12", 0],
                    "output_path": "./output",
                    "filename_prefix": "aps_retouched",
                    "extension": "png",
                    "quality": 100,
                    "overwrite_mode": "false",
                    "embed_workflow": "false" if self.strip_metadata else "true",
                },
            },
            "15": {
                "class_type": "UltralyticsDetectorProvider",
                "inputs": {"model_name": "face_yolov8m.pt"},
            },
        }

        payload = {"prompt": workflow, "client_id": client_id}
        log.info(
            "Workflow built: %d nodes, denoise=%.2f, seed=%d, client=%s",
            len(workflow), self.denoise, seed, client_id,
        )
        return payload

    def fire(self, image_path: str, mask_path: str) -> Dict[str, Any]:
        """
        Build workflow and submit to ComfyUI.
        If ComfyUI is offline, logs the error and returns the workflow payload
        so the architecture is preserved.

        Returns:
            {"status": "queued"|"offline", "prompt_id": ..., "payload": ...}
        """
        payload = self.build_workflow(image_path, mask_path)

        try:
            url = f"{self.api_url}/prompt"
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            prompt_id = resp.json().get("prompt_id", "unknown")
            log.info("Forge: queued prompt %s", prompt_id)
            return {"status": "queued", "prompt_id": prompt_id, "payload": payload}

        except requests.ConnectionError:
            log.warning(
                "Forge: ComfyUI connection refused at %s — server offline. "
                "Workflow payload preserved.", self.api_url
            )
            return {"status": "offline", "prompt_id": None, "payload": payload}

        except requests.RequestException as e:
            log.warning("Forge: ComfyUI unavailable (%s). Workflow payload preserved.", e)
            return {"status": "offline", "prompt_id": None, "payload": payload}

    def save_workflow_json(
        self, image_path: str, mask_path: str, output_file: str = "workflow.json"
    ) -> str:
        """Build and save the workflow JSON to disk for inspection/debugging."""
        payload = self.build_workflow(image_path, mask_path)
        with open(output_file, "w") as f:
            json.dump(payload, f, indent=2)
        log.info("Workflow saved → %s", output_file)
        return output_file


def run_forge(image_path: str, mask_path: str) -> Dict[str, Any]:
    """Convenience function: build Forge with defaults and fire."""
    forge = Forge()
    return forge.fire(image_path, mask_path)


if __name__ == "__main__":
    import sys

    img = sys.argv[1] if len(sys.argv) > 1 else "test_data/raws/fake_black.jpg"
    mask = sys.argv[2] if len(sys.argv) > 2 else "test_data/masks/fake_black_skin_mask.png"

    forge = Forge()

    # Save workflow JSON for inspection
    forge.save_workflow_json(img, mask, "test_data/workflow_debug.json")

    # Attempt to fire at ComfyUI
    result = forge.fire(img, mask)
    print(f"\nForge result: status={result['status']}")
    if result["prompt_id"]:
        print(f"  prompt_id={result['prompt_id']}")
    print(f"  workflow nodes: {len(result['payload']['prompt'])}")
