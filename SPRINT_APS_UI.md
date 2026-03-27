# Sprint 1: Auto Photo Studio (APS) Execution Dashboard

## Core Directives
Master requests a full visual interface and interaction design for the APS backend! This implies moving away from pure CLI and creating a web dashboard that complements the brutal aesthetics of RedIris.

### The Application Architecture
- We will build a **Next.js + Tailwind** lightweight Dashboard locally in `aps-dashboard` alongside the python engine.
- We will write a fast **FastAPI** layer in `aps_server.py` to wrap the `test_aps.py` and `forge` scripts so the web dashboard can trigger the pipeline and read progress.

### The UI Vision (Neo-noir Industrial Plant)
- **Primary Layout:** A multi-column dashboard mirroring a factory floor.
- **Column 1: The Input Hopper**
  - A massive drag-and-drop zone for ZIP folders or batches of raw images.
  - A live stream visual of "The Sorter" rejecting blurry images (Red glowing rejected queue).
- **Column 2: The Active Forge (Center Stage)**
  - Shows the image currently being processed by "The Mapper".
  - Two overlay views: Original Image vs Extracted Neural Mask (Cyberpunk tracking overlays).
- **Column 3: The Foundry Output**
  - A waterfall layout of finished, retouched images. Click to inspect the "Before/After" with a slider.

### Immediate Action Plan
1. Tell Master about this hardcore industrial UI concept.
2. Ask for permission to spin up a Next.js side-app + FastAPI backend in the workspace to make it real.
