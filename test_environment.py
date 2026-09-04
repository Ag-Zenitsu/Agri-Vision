import sys

def main():
    print("=" * 60)
    print(" Testing Python Library Installations (Windows) ")
    print("=" * 60)
    
    # 1. NumPy
    try:
        import numpy as np
        print(f"[PASS] NumPy version: {np.__version__}")
    except ImportError:
        print("[FAIL] NumPy is not installed.")
        
    # 2. Pandas
    try:
        import pandas as pd
        print(f"[PASS] Pandas version: {pd.__version__}")
    except ImportError:
        print("[FAIL] Pandas is not installed.")
        
    # 3. Matplotlib
    try:
        import matplotlib
        print(f"[PASS] Matplotlib version: {matplotlib.__version__}")
    except ImportError:
        print("[FAIL] Matplotlib is not installed.")
        
    # 4. OpenCV
    try:
        import cv2
        print(f"[PASS] OpenCV version: {cv2.__version__}")
    except ImportError:
        print("[FAIL] OpenCV is not installed.")
        
    # 5. TensorFlow
    try:
        import tensorflow as tf
        print(f"[PASS] TensorFlow version: {tf.__version__}")
        
        print("\n--- TensorFlow GPU Detection ---")
        # Check if TF was built with CUDA support
        if tf.test.is_built_with_cuda():
            print("[INFO] TensorFlow is built with CUDA support.")
        else:
            print("[INFO] TensorFlow is NOT built with CUDA support (CPU only).")
            
        # List physical GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[SUCCESS] GPU detected! {len(gpus)} GPU(s) available:")
            for gpu in gpus:
                print(f"  - {gpu}")
        else:
            print("[WARNING] No GPU detected. TensorFlow will run on CPU.")
            print("  * If you have an NVIDIA GPU, ensure CUDA Toolkit and cuDNN")
            print("    are installed and match your TensorFlow version requirements.")
            
    except ImportError:
        print("[FAIL] TensorFlow is not installed.")
    except Exception as e:
        print(f"[FAIL] TensorFlow failed to load: {e}")

    print("=" * 60)
    print(" Test completed. ")
    print("=" * 60)

if __name__ == "__main__":
    main()