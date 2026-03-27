"""
APS Darkroom Backend — FastAPI server wrapping the APS pipeline.
POST /api/process  →  accepts image upload, returns processed + mask URLs.
Serves /output as static files for the frontend to fetch results.
"""

import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from aps_local_test import ImageRecord, LocalSorter, LocalMapper

app = FastAPI(title="APS Darkroom API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Instantiate pipeline stages once at startup
sorter = LocalSorter()
mapper = LocalMapper()


@app.post("/api/process")
async def process_image(
    file: UploadFile = File(...),
    denoise: float = Form(0.30),
):
    """Accept an image, run Sorter → Mapper pipeline, return result URLs."""
    job_id = uuid.uuid4().hex[:12]
    ext = Path(file.filename or "image.png").suffix or ".png"

    # Save the uploaded original
    original_path = OUTPUT_DIR / f"{job_id}_original{ext}"
    contents = await file.read()
    with open(original_path, "wb") as f:
        f.write(contents)

    # Decode image for the pipeline
    img_array = np.frombuffer(contents, dtype=np.uint8)
    bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if bgr is None:
        original_path.unlink(missing_ok=True)
        return JSONResponse(
            status_code=400,
            content={"error": "Could not decode image"},
        )

    # Build an ImageRecord
    rec = ImageRecord(source_path=original_path)
    rec.decoded_bgr = bgr

    # Phase 1: Sorter — reject blurry / bad-exposure images
    sorter.evaluate(rec)
    if not rec.is_accepted:
        original_path.unlink(missing_ok=True)
        return JSONResponse(
            status_code=400,
            content={
                "error": "Image rejected by quality check",
                "quality_score": rec.quality_score,
                "sharpness": rec.sharpness_score,
                "exposure": rec.exposure_score,
            },
        )

    # Phase 2: Mapper — face detection + skin mask
    mapper.analyse(rec)

    # Save the skin mask as a PNG
    mask_path = OUTPUT_DIR / f"{job_id}_mask.png"
    cv2.imwrite(str(mask_path), rec.skin_mask)

    # Phase 3 placeholder: pass the original as the "processed" result
    processed_path = OUTPUT_DIR / f"{job_id}_processed{ext}"
    shutil.copy(original_path, processed_path)

    base = "/output"
    return JSONResponse(
        {
            "job_id": job_id,
            "denoise": denoise,
            "quality_score": rec.quality_score,
            "faces_detected": len(rec.face_boxes),
            "original": f"{base}/{original_path.name}",
            "processed": f"{base}/{processed_path.name}",
            "mask": f"{base}/{mask_path.name}",
        }
    )


# Serve the output directory as static files
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
