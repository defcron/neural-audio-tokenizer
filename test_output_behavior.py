#!/usr/bin/env python3
"""
Test script to validate the new output behavior requirements
"""
import subprocess
import sys
import tempfile
import os
from pathlib import Path

def run_tokenizer(args, input_file="test_simple.wav", timeout=20):
    """Run the tokenizer and capture stdout/stderr separately"""
    cmd = ["python", "neural_audio_tokenizer.py", "--compat-fallback"] + args + [input_file]
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True, 
        timeout=timeout
    )
    return result.returncode, result.stdout, result.stderr

def test_default_behavior():
    """Test: Default behavior should output NDJSON to stdout only"""
    print("🔬 Testing default behavior...")
    
    returncode, stdout, stderr = run_tokenizer([])
    
    # Should return success
    assert returncode == 0, f"Expected success, got return code {returncode}"
    
    # Should have NDJSON output in stdout
    assert '"event":"header"' in stdout, "Missing NDJSON header in stdout"
    assert '"event":"frame"' in stdout, "Missing NDJSON frames in stdout"  
    assert '"event":"end"' in stdout, "Missing NDJSON end marker in stdout"
    
    # Should have minimal stderr in default mode
    assert len(stderr.strip()) == 0, f"Expected clean stderr in default mode, got: {stderr[:100]}..."
    
    print("✅ Default behavior: PASS")

def test_verbose_behavior():
    """Test: Verbose mode should show NDJSON in stdout + verbose info in stderr"""
    print("🔬 Testing verbose behavior...")
    
    returncode, stdout, stderr = run_tokenizer(["--verbose"])
    
    # Should return success
    assert returncode == 0, f"Expected success, got return code {returncode}"
    
    # Should have NDJSON output in stdout
    assert '"event":"header"' in stdout, "Missing NDJSON header in stdout"
    assert '"event":"frame"' in stdout, "Missing NDJSON frames in stdout"
    assert '"event":"end"' in stdout, "Missing NDJSON end marker in stdout"
    
    # Should have verbose info in stderr
    assert "Enhanced Neural Audio-to-LLM Tokenizer" in stderr, "Missing verbose info in stderr"
    assert "INFO" in stderr, "Missing INFO messages in stderr"
    
    print("✅ Verbose behavior: PASS")

def test_v_shorthand():
    """Test: -v shorthand flag should work like --verbose"""
    print("🔬 Testing -v shorthand...")
    
    returncode, stdout, stderr = run_tokenizer(["-v"])
    
    # Should return success
    assert returncode == 0, f"Expected success, got return code {returncode}"
    
    # Should have NDJSON and verbose output like --verbose
    assert '"event":"header"' in stdout, "Missing NDJSON header in stdout"
    assert "Enhanced Neural Audio-to-LLM Tokenizer" in stderr, "Missing verbose info in stderr"
    
    print("✅ -v shorthand: PASS")

def test_explicit_ndjson_streaming():
    """Test: Explicit --ndjson-streaming should work with stream locking"""
    print("🔬 Testing explicit --ndjson-streaming...")
    
    returncode, stdout, stderr = run_tokenizer(["--ndjson-streaming", "--log-level", "INFO"])
    
    # Should return success
    assert returncode == 0, f"Expected success, got return code {returncode}"
    
    # Should have NDJSON output in stdout
    assert '"event":"header"' in stdout, "Missing NDJSON header in stdout"
    
    # May have info in stderr, but NDJSON should be clean due to stream locking
    lines = stdout.strip().split('\n')
    for line in lines:
        if line.strip():  # Skip empty lines
            assert line.startswith('{') and line.endswith('}'), f"Invalid NDJSON line: {line[:50]}..."
    
    print("✅ Explicit --ndjson-streaming: PASS")

def test_error_handling():
    """Test: Errors should go to stderr, stdout should remain clean"""
    print("🔬 Testing error handling...")
    
    returncode, stdout, stderr = run_tokenizer([], input_file="nonexistent.wav")
    
    # Should return error
    assert returncode != 0, "Expected error for nonexistent file"
    
    # stdout should be empty
    assert len(stdout.strip()) == 0, f"Expected clean stdout on error, got: {stdout[:100]}..."
    
    # stderr should contain error info
    assert "FileNotFoundError" in stderr or "RuntimeError" in stderr, "Missing error info in stderr"
    
    print("✅ Error handling: PASS")

def test_log_levels():
    """Test: Different log levels should show appropriate output"""
    print("🔬 Testing log levels...")
    
    # ERROR level - should be very minimal
    _, stdout_error, stderr_error = run_tokenizer(["--log-level", "ERROR"])
    
    # WARN level (default) - should be clean  
    _, stdout_warn, stderr_warn = run_tokenizer(["--log-level", "WARN"])
    
    # INFO level - should be verbose
    _, stdout_info, stderr_info = run_tokenizer(["--log-level", "INFO"])
    
    # All should have NDJSON in stdout
    for stdout in [stdout_error, stdout_warn, stdout_info]:
        assert '"event":"header"' in stdout, "Missing NDJSON in stdout"
    
    # ERROR level should have least stderr output
    # INFO level should have most stderr output  
    assert len(stderr_info) > len(stderr_warn), "INFO should have more stderr than WARN"
    
    print("✅ Log levels: PASS")

def main():
    """Run all tests"""
    print("🧪 Testing Neural Audio Tokenizer Output Behavior")
    print("=" * 50)
    
    # Ensure test file exists
    if not Path("test_simple.wav").exists():
        print("❌ test_simple.wav not found - creating...")
        # Create a simple test file
        import numpy as np
        try:
            import soundfile as sf
            sample_rate = 22050
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * 440 * t) * 0.5
            sf.write('test_simple.wav', audio, sample_rate)
            print("✅ Created test_simple.wav")
        except ImportError:
            print("❌ Cannot create test file - soundfile not available")
            return 1
    
    try:
        test_default_behavior()
        test_verbose_behavior()  
        test_v_shorthand()
        test_explicit_ndjson_streaming()
        test_error_handling()
        test_log_levels()
        
        print("\n" + "=" * 50)
        print("🎉 All tests PASSED!")
        print("✅ Default behavior: Raw NDJSON to stdout")
        print("✅ Verbose flags: -v/--verbose work correctly")  
        print("✅ Stream locking: NDJSON integrity maintained")
        print("✅ Error handling: Errors go to stderr, stdout clean")
        print("✅ Log levels: Appropriate verbosity control")
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())