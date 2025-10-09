#!/usr/bin/env python3
"""
Test script to verify all requirements.txt dependencies can be imported
"""

def test_core_dependencies():
    """Test core ML and numerical dependencies"""
    try:
        import numpy as np
        import torch
        import torchaudio
        print("✅ Core ML dependencies: OK")
        return True
    except ImportError as e:
        print(f"❌ Core ML dependencies failed: {e}")
        return False

def test_audio_dependencies():
    """Test audio processing dependencies"""
    try:
        import librosa
        import soundfile as sf
        print("✅ Audio processing dependencies: OK")
        return True
    except ImportError as e:
        print(f"❌ Audio processing dependencies failed: {e}")
        return False

def test_ml_dependencies():
    """Test machine learning dependencies"""
    try:
        import transformers
        import scipy
        import sklearn
        print("✅ ML dependencies: OK")
        return True
    except ImportError as e:
        print(f"❌ ML dependencies failed: {e}")
        return False

def test_visualization_dependencies():
    """Test visualization dependencies"""
    try:
        import matplotlib
        import seaborn
        print("✅ Visualization dependencies: OK")
        return True
    except ImportError as e:
        print(f"❌ Visualization dependencies failed: {e}")
        return False

def test_optional_dependencies():
    """Test optional dependencies"""
    try:
        import encodec
        import psutil
        print("✅ Optional dependencies: OK")
        return True
    except ImportError as e:
        print(f"❌ Optional dependencies failed: {e}")
        return False

def main():
    """Run all dependency tests"""
    print("🧪 Testing Neural Audio Tokenizer Requirements")
    print("=" * 50)
    
    all_tests = [
        test_core_dependencies,
        test_audio_dependencies, 
        test_ml_dependencies,
        test_visualization_dependencies,
        test_optional_dependencies
    ]
    
    results = []
    for test in all_tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    if all(results):
        print("🎉 All dependencies are working correctly!")
        print("✅ requirements.txt is complete and functional")
        return 0
    else:
        print("❌ Some dependencies are missing or not working")
        return 1

if __name__ == "__main__":
    exit(main())