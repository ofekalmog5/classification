# Image Material Classification (Desktop)

Desktop app for unsupervised classification of ortho imagery using KMeans clustering.

## Run Tkinter app (PowerShell)
1. Create venv and install deps:
   - `py -3.11 -m venv .venv`
   - `.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt`
2. Start Tkinter UI:
   - `.\.venv\Scripts\python.exe tkinter_app.py`

## Run backend API (optional)
If you still want the API server:
- `.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --app-dir backend`

Backend default address: 127.0.0.1:8000

## Run Electron app (optional)
Requires Node.js + npm:
- `cd app`
- `npm install`
- `npm run dev`

## How It Works
- **Unsupervised Classification**: Uses KMeans clustering to automatically classify the entire image into the specified number of material classes
- **No Training Required**: Classification is done automatically based on spectral features - no training samples needed
- **Vector Overlay**: Vector layers (optional) are drawn on top of the result in yellow/orange for marking areas of interest
- **Smooth Results**: Superpixels segmentation creates smooth, homogeneous regions

## Notes
- Output GeoTIFF is saved next to the input raster with a *_classified.tif suffix
- Select the number of materials (clusters) before running
- NDVI uses fixed bands (red=3, NIR=4) and is available only in multispectral mode
- Class colors are auto-generated from the number of materials selected
- Vector layers are optional and will be overlaid in yellow on the final result
