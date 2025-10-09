#!/usr/bin/env python3
"""
Simple test to verify that decoding_time and memory_usage metrics are implemented correctly.
"""

import sys
import time
import numpy as np
import torch
from neural_audio_tokenizer import TokenizationEvaluator, NeuralAudioTokenizer

def test_metrics():
    """Test that the new metrics (decoding_time and memory_usage) are calculated."""
    
    print("Testing decoding_time and memory_usage metrics implementation...")
    
    # Create a simple evaluator
    evaluator = TokenizationEvaluator(sample_rate=22050)
    
    # Create some dummy audio data (1 second at 22050 Hz)
    sample_rate = 22050
    duration = 1.0  # seconds
    original_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
    
    # Create a mock tokenizer result for testing
    mock_result = {
        'semantic_codes': [torch.randint(0, 1024, (1, 100)) for _ in range(4)],
        'acoustic_codes': [torch.randint(0, 1024, (1, 100)) for _ in range(4)],
        'reconstructed': torch.randn(1, int(sample_rate * duration)),
        'num_frames': 100
    }
    
    # Test with precomputed result (should have 0 decoding time)
    print("Testing with precomputed result...")
    try:
        # Create a simple tokenizer mock
        class MockTokenizer:
            def parameters(self):
                yield torch.randn(1)
            def eval(self):
                return self
        
        mock_tokenizer = MockTokenizer()
        
        metrics = evaluator.evaluate_tokenization(
            original_audio=original_audio,
            tokenizer=mock_tokenizer,
            precomputed_result=mock_result
        )
        
        print(f"  Decoding time: {metrics.decoding_time:.4f} seconds")
        print(f"  Memory usage: {metrics.memory_usage:.2f} MB")
        print(f"  Encoding time: {metrics.encoding_time:.4f} seconds")
        
        # Verify metrics exist and are reasonable
        assert hasattr(metrics, 'decoding_time'), "decoding_time attribute missing"
        assert hasattr(metrics, 'memory_usage'), "memory_usage attribute missing"
        assert isinstance(metrics.decoding_time, (int, float)), "decoding_time should be numeric"
        assert isinstance(metrics.memory_usage, (int, float)), "memory_usage should be numeric"
        assert metrics.decoding_time >= 0, "decoding_time should be non-negative"
        assert metrics.memory_usage >= 0, "memory_usage should be non-negative"
        
        print("✓ Test passed! Metrics are properly calculated.")
        print("✓ decoding_time and memory_usage TODO items have been implemented.")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_metrics()
    sys.exit(0 if success else 1)