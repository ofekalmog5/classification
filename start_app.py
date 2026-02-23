#!/usr/bin/env python3
import subprocess
import sys

print("=" * 70)
print("CLASSIFICATION APP LAUNCHER")
print("=" * 70)

try:
    # Try to run the tkinter app
    print("\nAttempting to launch tkinter app...")
    result = subprocess.run(
        [sys.executable, "tkinter_app.py"],
        cwd="C:\\Users\\ofeka\\OneDrive\\Desktop\\Jobs\\mil\\classification"
    )
    sys.exit(result.returncode)
except Exception as e:
    print(f"\nError launching app: {e}")
    print(f"\nTrying to show error details...")
    
    # Try importing to show what's wrong
    try:
        from backend.app.core import classify
        print("✓ Core module imported successfully!")
    except ImportError as ie:
        print(f"✗ Import error: {ie}")
    except Exception as ex:
        print(f"✗ Other error: {ex}")
        import traceback
        traceback.print_exc()
    
    sys.exit(1)
