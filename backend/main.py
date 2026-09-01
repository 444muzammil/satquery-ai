from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import rasterio
import numpy as np
from PIL import Image, ImageDraw
import io
import base64
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VQARequest(BaseModel):
    image_base64: str
    query: str
    width: int = 500  # Default fallback
    height: int = 500

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    content = await file.read()
    is_tiff = file.filename.lower().endswith(('.tif', '.tiff', '.geotiff'))
    metadata = {"filename": file.filename, "format": "GeoTIFF/TIFF" if is_tiff else "PNG/JPEG"}
    preview_base64 = ""

    if is_tiff:
        with rasterio.MemoryFile(content) as memfile:
            with memfile.open() as dataset:
                metadata.update({
                    "width": dataset.width, "height": dataset.height,
                    "bands": dataset.count, "crs": str(dataset.crs) if dataset.crs else "Unspecified",
                    "resolution": dataset.res
                })
                bands_to_read = min(3, dataset.count)
                img_array = dataset.read(list(range(1, bands_to_read + 1)))
                if bands_to_read == 1:
                    img_array = img_array[0]
                else:
                    img_array = np.moveaxis(img_array, 0, -1)
                img_array = (255 * (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array) + 1e-8)).astype(np.uint8)
                img = Image.fromarray(img_array)
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                preview_base64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
    else:
        img = Image.open(io.BytesIO(content))
        metadata.update({
            "width": img.width, "height": img.height, "bands": len(img.getbands()),
            "crs": "N/A (Standard Image)", "resolution": "N/A"
        })
        preview_base64 = f"data:image/{img.format.lower()};base64,{base64.b64encode(content).decode()}"

    return {"metadata": metadata, "preview": preview_base64}


@app.post("/api/vqa")
async def analyze_image(request: VQARequest):
    query = request.query.lower()
    mask_base64 = None
    
    # Visual Grounding Logic: Triggered by "where" or "highlight"
    if "where" in query or "highlight" in query:
        # Create a blank transparent image matching original dimensions
        mask = Image.new('RGBA', (request.width, request.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(mask)
        
        # Simulate model detecting a specific region (e.g., water or buildings)
        color = (0, 150, 255, 128) if "water" in query else (255, 50, 50, 128)
        
        # Draw a mock bounding box / mask area
        box = [request.width * 0.2, request.height * 0.4, request.width * 0.6, request.height * 0.8]
        draw.rectangle(box, fill=color, outline=(255, 255, 255, 255), width=3)
        
        buffered = io.BytesIO()
        mask.save(buffered, format="PNG")
        mask_base64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
        
        answer = "I have highlighted the requested features in the viewer."
        confidence = 94
        trace = "Agent -> Grounding Model Selected -> Region Mask Generated"
        evidence_text = "Bounding Box: [x1:0.2, y1:0.4, x2:0.6, y2:0.8]"
        
    elif "water" in query:
        answer = "The image contains significant water bodies, likely a river or coastal feature."
        confidence = 88
        trace = "Agent -> Single-Image VQA Executed"
        evidence_text = "VLM Text Generation"
    else:
        answer = f"Analysis complete for query: '{request.query}'. Land cover features identified."
        confidence = random.randint(75, 92)
        trace = "Agent -> Single-Image VQA Executed"
        evidence_text = "VLM Text Generation"
        
    return {
        "answer": answer,
        "confidence": confidence,
        "trace": trace,
        "mask": mask_base64,
        "evidence": evidence_text
    }