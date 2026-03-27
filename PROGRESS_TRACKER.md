# Auto Photo Studio (APS) Backend Tracker

## Phase 1: Real MediaPipe Mapper ✅
- Native Python Mediapipe segmentation. Extracts separate skin and BG masks.

## Phase 2: PIQE Sorter ✅
- Laplacian Variance blur detection mechanism to reject OOF frames directly.

## Phase 3: The Headless Forge ✅
- Connects JSON payload via WebSocket to a latent pipeline. Locked `denoise` 0.25 - 0.35.

## Sprint: The Darkroom (Commercial UI & FastAPI Backend) ✅
- Implemented `aps_server.py` FastAPI backend with Image Pipeline parsing, JSON output and static mounts.
- Next.js Tailwinds `#0a0a0a` industrial dashboard built and linked.
- Drag-and-drop Image Hopper, 3-layer masking inspector, strict hotkeys (Space & M).
- Successfully compiles at 0 Typescript Errors.
