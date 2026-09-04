import rasterio
from rasterio.transform import from_origin
import numpy as np

# Coordinates mapping to Nashik, Maharashtra
transform = from_origin(73.78, 19.99, 0.0001, 0.0001) 
crs = "EPSG:4326"
width, height = 512, 512

# 1. OPTICAL IMAGE A (Base)
opt_a = np.zeros((3, height, width), dtype=np.uint8)
opt_a[0, :, :] = 34   # R
opt_a[1, :, :] = 139  # G (Green land)
opt_a[2, :, :] = 34   # B
# Draw a river
opt_a[0, 200:300, :] = 25 
opt_a[1, 200:300, :] = 100
opt_a[2, 200:300, :] = 220 

with rasterio.open('test_optical_A.tif', 'w', driver='GTiff', height=height, width=width, count=3, dtype=str(opt_a.dtype), crs=crs, transform=transform) as f:
    f.write(opt_a)

# 2. OPTICAL IMAGE B (Bi-temporal Change)
opt_b = opt_a.copy()
# Simulate a flood by widening the river
opt_b[0, 180:320, :] = 40 
opt_b[1, 180:320, :] = 80
opt_b[2, 180:320, :] = 200

with rasterio.open('test_optical_B.tif', 'w', driver='GTiff', height=height, width=width, count=3, dtype=str(opt_b.dtype), crs=crs, transform=transform) as f:
    f.write(opt_b)

# 3. SAR IMAGE (Cross-Modal Radar)
# 16-bit array. Water is dark (low backscatter), land is medium, buildings are bright (double-bounce).
sar = np.random.normal(15000, 3000, (1, height, width)).astype(np.uint16)
sar[0, 200:300, :] = np.random.normal(2000, 500, (100, width)) # Dark specular water
sar[0, 100:150, 100:150] = np.random.normal(60000, 2000, (50, 50)) # Bright urban structure

with rasterio.open('test_sar.tif', 'w', driver='GTiff', height=height, width=width, count=1, dtype=str(sar.dtype), crs=crs, transform=transform) as f:
    f.write(sar)

print("Generated test_optical_A.tif, test_optical_B.tif, and test_sar.tif!")