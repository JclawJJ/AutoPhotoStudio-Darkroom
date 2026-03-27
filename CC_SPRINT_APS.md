# APS Backend Implementation Directive (CC Action Plan)

The frontend is paused. We must now turn `APS_Project/` from a conceptual outline into a **brutal, fully functional Python backend**. Master demands real implementation, audit, and tests of the Auto Photo Studio.

## Mission Execution Targets

### Phase 1: Real MediaPipe Mapper (Masking)
Write a Python module (`mapper.py` or within `aps_pipeline.py`) that genuinely imports `mediapipe`, load a test image, extracts the high-quality semantic masks for (1) Face/Skin and (2) Background. Save mask image structures to `/Users/jclaw/.openclaw/workspace/APS_Project/test_data/masks/`. 

### Phase 2: The Blurs Sorter (PIQE / Laplacian)
Create `sorter.py`. Read images from `/Users/jclaw/.openclaw/workspace/APS_Project/test_data/raws/`. Use OpenCV Laplacian Variance filter to calculate standard blur deviation. If the variance is too low (blurry), move it to `/Users/jclaw/.openclaw/workspace/APS_Project/test_data/rejected/`.

### Phase 3: The Forge (ComfyUI WebSocket Execution)
Create `forge.py`. It should contain the logic to assemble a ComfyUI JSON workflow payload (with `denoise` locked aggressively to 0.25 - 0.35 to prevent AI plastic skin). The payload points to the original image and the mask we made. It fires it headlessly to `http://127.0.0.1:8188` (ComfyUI default).  (If Comfy is offline, log connection refused but maintain the logic architecture flawlessly).

### Phase 4: Full Local Audit
Provide a dry-run test script `test_aps.py`.
Run it. Handle all missing pip dependencies like `opencv-python`, `mediapipe`, `requests`.

We are building a commercial retouching weapon. NO HALLUCINATION placeholders like `# do stuff here`. Write the real algorithmic logic.
