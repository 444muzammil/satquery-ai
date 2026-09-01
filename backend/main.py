from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import rasterio
import numpy as np
from PIL import Image
import io
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        # Process GeoTIFF/TIFF for geospatial imagery
        with rasterio.MemoryFile(content) as memfile:
            with memfile.open() as dataset:
                metadata.update({
                    "width": dataset.width,
                    "height": dataset.height,
                    "bands": dataset.count,
                    "crs": str(dataset.crs) if dataset.crs else "Unspecified",
                    "resolution": dataset.res
                })
                
                # Create a visual preview (extract first 3 bands for RGB, or just band 1 if grayscale)
                bands_to_read = min(3, dataset.count)
                img_array = dataset.read(list(range(1, bands_to_read + 1)))
                
                # Normalize array for PNG conversion
                if bands_to_read == 1:
                    img_array = img_array[0]
                else:
                    img_array = np.moveaxis(img_array, 0, -1) # Convert (C, H, W) to (H, W, C)
                
                # Scale to 0-255 for display
                img_array = (255 * (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array) + 1e-8)).astype(np.uint8)
                
                img = Image.fromarray(img_array)
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                preview_base64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
    else:
        # Handle PNG/JPEG inputs for public benchmark datasets
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