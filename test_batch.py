import requests
import json

url = "http://127.0.0.1:8000/classify-batch"
payload = {
    "rasterPaths": ["test_raster.tif"],
    "classes": [{"id": "1", "name": "BM_ASPHALT", "color": "#2D2D30"}],
    "vectorLayers": [],
    "smoothing": "none",
    "featureFlags": {
        "spectral": True,
        "texture": False,
        "indices": False,
        "colorIndices": True,
        "entropy": False,
        "morphCleanup": True
    }
}

try:
    response = requests.post(url, json=payload)
    print("Status Code:", response.status_code)
    print("Response:", response.text)
except Exception as e:
    print("Error:", e)
