#!/usr/bin/env python3
"""
Setup and test script for the refactored classification pipeline.
Installs dependencies and runs validation tests.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and report results"""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"$ {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print(f"\n✓ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} - FAILED")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"\n✗ {description} - ERROR")
        print(f"Error: {e}")
        return False


def main():
    root = Path(__file__).parent
    
    print("\n" + "="*70)
    print("CLASSIFICATION PIPELINE REFACTORING - SETUP & VALIDATION")
    print("="*70)
    
    # Step 1: Check Python
    print("\nStep 1: Checking Python environment...")
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    
    # Step 2: Install dependencies
    print("\nStep 2: Installing dependencies...")
    requirements_file = root / "backend" / "requirements.txt"
    
    if requirements_file.exists():
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        if not run_command(cmd, "Install from requirements.txt"):
            print("\nWarning: Some dependencies may not have installed")
    
    # Step 3: Compile Python files
    print("\nStep 3: Validating Python syntax...")
    files_to_compile = [
        "backend/app/main.py",
        "backend/app/core.py"
    ]
    
    all_compiled = True
    for file in files_to_compile:
        cmd = [sys.executable, "-m", "py_compile", file]
        if not run_command(cmd, f"Compile {file}"):
            all_compiled = False
    
    if not all_compiled:
        print("\n✗ Some files have syntax errors")
        return False
    
    # Step 4: Test imports
    print("\nStep 4: Testing imports...")
    try:
        sys.path.insert(0, str(root / "backend"))
        from app.core import classify, classify_and_export, rasterize_vectors_onto_classification
        print("✓ Successfully imported all core functions")
        
        from app.main import app
        print("✓ Successfully imported FastAPI app")
        
        print("\n" + "="*70)
        print("✓ ALL VALIDATION TESTS PASSED")
        print("="*70)
        print("\nRefactored classification pipeline is ready for testing!")
        print("\nAvailable endpoints:")
        print("  - POST /classify              (Full pipeline)")
        print("  - POST /classify-step1        (KMeans only)")
        print("  - POST /classify-step2        (Vector rasterization only)")
        print("  - GET  /health                (Health check)")
        
        print("\nTo start the API server:")
        print("  $ uvicorn backend.app.main:app --reload --port 8000")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("\nMake sure all dependencies are installed:")
        print("  $ pip install fastapi uvicorn rasterio geopandas scikit-learn numpy scipy")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
