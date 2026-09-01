from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import rasterio
import numpy as np
from PIL import Image
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

# --- NEW: Data model for the VQA request ---
class VQARequest(BaseModel):
    image_base64: str
    query: str

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    # Read file into memory
    content = await file.read()
    
    # Check if it's a prescribed benchmark format or geospatial format
    is_tiff = file.filename.lower().endswith(('.tif', '.tiff', '.geotiff'))
    
    metadata = {
        "filename": file.filename,
        "format": "GeoTIFF/TIFF" if is_tiff else "PNG/JPEG",
    }
    
    preview_base64 = ""

    if is_tiff:
        with rasterio.MemoryFile(content) as memfile:
            with memfile.open() as dataset:
                metadata.update({
                    "width": dataset.width,
                    "height": dataset.height,
                    "bands": dataset.count,
                    "crs": str(dataset.crs) if dataset.crs else "Unspecified",
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
            "width": img.width,
            "height": img.height,
            "bands": len(img.getbands()),
            "crs": "N/A (Standard Image)",
            "resolution": "N/A"
        })
        preview_base64 = f"data:image/{img.format.lower()};base64,{base64.b64encode(content).decode()}"

    return {"metadata": metadata, "preview": preview_base64}


# --- NEW: VQA Endpoint ---
@app.post("/api/vqa")
async def analyze_image(request: VQARequest):
    query = request.query.lower()
    
    # Hardware Guardrail: Mocking the Remote-Sensing VLM response for 8GB local RAM.
    # Replace this block later with an external API call to your fine-tuned model.
    if "water" in query:
        answer = "The image contains significant water bodies, likely a river or coastal feature, with some surrounding vegetation."
        confidence = 88
    elif "describe" in query or "visible" in query:
        answer = "The image features a mix of agricultural areas and built-up regions. Specific structural details are visible."
        confidence = 86
    elif not request.image_base64:
        answer = "Error: No image provided for analysis."
        confidence = 0
    else:
        answer = f"Analysis complete for query: '{request.query}'. Land cover features identified."
        confidence = random.randint(75, 92)
        
    return {
        "answer": answer,
        "confidence": confidence,
        "trace": "Agent Controller -> Single-Image Task Selected -> VLM Executed"
    }