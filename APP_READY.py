#!/usr/bin/env python3
"""
Show the updated tkinter app with two-step classification
"""

print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║   TKINTER APP - TWO-STEP CLASSIFICATION ARCHITECTURE              ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝

✓ APP UPDATED WITH THREE BUTTONS:

1️⃣  "Step 1: Run Classification"
   └─ Runs: classify_and_export()
   └─ Output: RGB GeoTIFF (classification only, no vectors)
   └─ Use case: Fast classification, adjust vectors later

2️⃣  "Step 2: Rasterize Vectors"  
   └─ Runs: rasterize_vectors_onto_classification()
   └─ Input: Classification file from Step 1
   └─ Output: RGB GeoTIFF + vector overlay
   └─ Use case: Apply vectors to existing classification

3️⃣  "Full Pipeline"
   └─ Runs: classify() wrapper (both steps)
   └─ Output: Complete result
   └─ Use case: One-click classification with vectors

───────────────────────────────────────────────────────────────────

📋 FILE CHANGES:

✓ backend/app/core.py
  - classify() → wrapper function (45 lines)
  - classify_and_export() → Step 1 isolated (155 lines)
  - rasterize_vectors_onto_classification() → Step 2 isolated (160 lines)

✓ backend/app/main.py
  - POST /classify-step1 → Call classify_and_export()
  - POST /classify-step2 → Call rasterize_vectors_onto_classification()
  - POST /classify → Full pipeline (backward compatible)

✓ tkinter_app.py
  - Updated imports (added new functions)
  - New buttons: Step 1, Step 2, Full Pipeline
  - Separated logic for each step
  - Independent threads for each operation

───────────────────────────────────────────────────────────────────

🚀 HOW TO RUN:

Option A: Direct Python
  python tkinter_app.py

Option B: Via Batch
  run_app.bat

Option C: Via Launcher
  python launcher.py

Option D: Via API (no GUI)
  uvicorn backend.app.main:app --reload --port 8000
  
  Then use /docs for Swagger UI or make POST requests to:
  - /classify-step1
  - /classify-step2
  - /classify

───────────────────────────────────────────────────────────────────

💡 WORKFLOW EXAMPLE:

Step 1: Load raster → Set classes → Click "Step 1"
  ✓ Output: classification.tif

Step 2: Load vectors → Click "Step 2"
  ✓ Input: classification.tif
  ✓ Output: classification_with_vectors.tif

OR: Click "Full Pipeline" to do both at once

───────────────────────────────────────────────────────────────────

✓ All files compile without errors
✓ No missing dependencies (require installation)
✓ Backward compatible with existing code
✓ Ready for testing

═══════════════════════════════════════════════════════════════════
""")
