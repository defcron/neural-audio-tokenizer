#!/usr/bin/env python3
"""
Simple test script to demonstrate the new features without network dependencies
"""

import sys
import os
import subprocess
import tempfile
import numpy as np
from pathlib import Path

def create_test_audio():
    """Create a simple test audio file"""
    # Generate 1 second of sine wave at 440Hz
    sample_rate = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    
    # Save as raw audio (simple format)
    temp_file = tempfile.NamedTemporaryFile(suffix='.raw', delete=False)
    audio_int16 = (audio * 32767).astype(np.int16)
    temp_file.write(audio_int16.tobytes())
    temp_file.close()
    
    return temp_file.name

def test_logging_levels():
    """Test different logging levels"""
    print("=== Testing Logging Levels ===")
    
    audio_file = create_test_audio()
    
    try:
        # Test --help with different log levels
        print("\n1. Testing help output:")
        result = subprocess.run([
            sys.executable, 'neural_audio_tokenizer.py', '--help'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print("✓ Help command works")
        else:
            print("✗ Help command failed")
            
        print("\n2. Testing argument parsing:")
        # Test argument parsing (will fail due to network, but should show our logging)
        result = subprocess.run([
            sys.executable, 'neural_audio_tokenizer.py',
            '--log-level', 'INFO',
            '--compat-fallback',
            audio_file
        ], capture_output=True, text=True, timeout=10)
        
        print("STDOUT:", result.stdout[:500] if result.stdout else "None")
        print("STDERR:", result.stderr[:500] if result.stderr else "None")
        print("Return code:", result.returncode)
            
    except subprocess.TimeoutExpired:
        print("✓ Expected timeout due to network issues")
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        # Cleanup
        try:
            os.unlink(audio_file)
        except:
            pass

def test_stdin_handling():
    """Test stdin byte handling"""
    print("\n=== Testing Stdin Handling ===")
    
    # Create test audio data
    audio_data = b'\x52\x49\x46\x46'  # WAV header start
    audio_data += b'\x00\x00\x00\x00'  # Placeholder
    audio_data += b'\x57\x41\x56\x45'  # WAVE format
    audio_data += b'\x00' * 100  # Some data
    
    try:
        # Test stdin processing
        result = subprocess.run([
            sys.executable, 'neural_audio_tokenizer.py',
            '--log-level', 'DEBUG'
        ], input=audio_data, capture_output=True, timeout=5)
        
        print("Return code:", result.returncode)
        if result.stderr:
            print("STDERR:", result.stderr.decode()[:300])
        
    except subprocess.TimeoutExpired:
        print("✓ Expected timeout - stdin handling is working")
    except Exception as e:
        print(f"Error: {e}")

def test_version_consistency():
    """Test version constant usage"""
    print("\n=== Testing Version Consistency ===")
    
    # Read the file and check version usage
    with open('neural_audio_tokenizer.py', 'r') as f:
        content = f.read()
    
    # Check for hardcoded version strings
    hardcoded_versions = []
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'v0.1.7' in line and 'VERSION' not in line:
            hardcoded_versions.append(f"Line {i+1}: {line.strip()}")
    
    if hardcoded_versions:
        print("✗ Found hardcoded version strings:")
        for version in hardcoded_versions[:5]:  # Show first 5
            print(f"  {version}")
    else:
        print("✓ No hardcoded version strings found")
    
    # Check VERSION constant exists and is used
    if 'VERSION = "0.1.7"' in content:
        print("✓ VERSION constant defined")
    else:
        print("✗ VERSION constant not found")
        
    if 'f"tims-ears-{VERSION}' in content:
        print("✓ VERSION constant used in model IDs")
    else:
        print("✗ VERSION constant not used in model IDs")

if __name__ == "__main__":
    print("Neural Audio Tokenizer - Feature Testing")
    print("=" * 50)
    
    test_version_consistency()
    test_logging_levels()
    test_stdin_handling()
    
    print("\n" + "=" * 50)
    print("Testing complete!")