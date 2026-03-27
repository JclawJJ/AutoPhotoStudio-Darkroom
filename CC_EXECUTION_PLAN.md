# APS_Project (Auto Photo Studio) - Execution Thread by Claude Code (CC)

## Project Overview
APS is an automated batch photo-retouching pipeline tailored for Sony A7M5 raw/jpg files.
**Goal:** Deliver commercial-grade portrait retouching (Evoto / Pixel Cake equivalents) via a zero-UI, pure script-driven backend.
**Aesthetic Standard:** Frequency Separation (preserve high-freq pores/texture, smooth low-freq shadows), Neutral Gray (Dodge & Burn) for 3D contouring, NO "plastic/silicone" fake AI masks. Eastern natural aesthetic.

## 🚀 Commercial Upgrade Blueprint (v2026.3)
Based on top-tier architectural decisions and deep-dive into photography skills:

1. **The Sorter (Rank-IQA/PIQE):** Python script to filter out blurry/bad exposures.
2. **The Mapper (Semantic + Instance Decoupling):**
   - Decouple facial skin processing from highlight preservation.
   - Cache masked embeddings per image to allow sub-50ms UI manipulation.
   - YOLOv8 + MediaPipe (SelfieSegmentation + FaceMesh) for granular element extraction.
   - ~~BiSeNet-V2 removed~~ — 79999_iter.pth download URL returned 404. Replaced with fully-local MediaPipe solution (no weight downloads required).
3. **The Forge (ComfyUI Headless CLI + WAS Node Suite + Level Pixel Nodes):** 
   - **Environment:** Must utilize WAS Node Suite and Level Pixel Nodes for native noise alignment and highres pixel-level inpainting.
   - **Security:** Strict IP lock. No metadata/workflow JSON leaked to the output PNG.
   - **System Prompt (Directives):** `"lightly even out skin tone and remove redness, preserve natural skin texture and facial details, asian aesthetic, macro photography, highly detailed"`
   - **Negative Constraints:** `"glamour retouching, heavy makeup, plastic skin, silicone reflection, glossy face, low detail"`
   - **Parameters:** `denoise` constrained strictly to 0.25-0.35.

## CC's Task List
- [x] Initialize Python environment natively.
- [x] Write REAL implementation of `aps_pipeline.py` (no placeholders).
- [x] Implement The Sorter (Image Quality Assessment).
- [x] Implement The Mapper (Face - [ ] Implement The Mapper (Face & Skin segmentation logic + caching). Skin segmentation logic + caching).
- [x] Write the ComfyUI headless JSON executor for The Forge.
- [x] Test end-to-end on a dummy directory of 5 images.
- [x] Replace BiSeNet-V2 (404 weights URL) with MediaPipe SelfieSegmentation + FaceMesh.
- [x] Add `--mode convert` CLI for bulk image format conversion (sprint2_extensions item).