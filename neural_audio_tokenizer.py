#!/usr/bin/env python3
"""
neural_audio_tokenizer.py - By Claude Sonnet 4 (Extended Thinking Mode), based on initial work by ChatGPT Agent Mode, and with help from custom GPT Tuesday, GPT-5 Auto, and Jeremy Carter <jeremy@jeremycarter.ca> - 2025-09-16
==========================

A research-grade neural audio tokenization system optimized for LLM consumption,
specifically designed for music understanding. Implements state-of-the-art 
hybrid tokenization strategies based on recent advances in AudioLM, MuQ, 
SoundStream, and other neural codec research.

Key Features:
- Hybrid semantic + acoustic tokenization
- Neural codec with residual vector quantization  
- Music-specific representation learning
- Multi-scale temporal modeling
- Scientific evaluation with reconstruction metrics
- Iterative optimization for LLM-optimal representations
- Streaming output protocols for large-scale processing

Based on:
- AudioLM: Language Modeling Approach to Audio Generation (Borsos et al., 2022)
- MuQ: Self-Supervised Music Representation with Mel-RVQ (Zhu et al., 2025)
- SoundStream: End-to-End Neural Audio Codec (Zeghidour et al., 2021)
- CLaM-TTS: Neural Codec Language Model improvements (Kim et al., 2024)
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import base64
from pathlib import Path
import tempfile
import gc

# Core numerical and ML libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.transforms as T

# Traditional audio processing (for baselines and evaluation)
import librosa
import soundfile as sf
from scipy import signal
from scipy.stats import entropy
import sklearn.metrics

# Optional visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Optional advanced features
try:
    import transformers
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from encodec import EncodecModel
    HAS_ENCODEC = True
except ImportError:
    HAS_ENCODEC = False

warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# Core Neural Architecture Components
# ============================================================================

class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer based on SoundStream and MuQ architectures.
    Implements hierarchical quantization for efficient music representation.
    """
    def __init__(self, 
                 input_dim: int = 512,
                 codebook_size: int = 1024,
                 num_quantizers: int = 8,
                 commitment_weight: float = 0.25,
                 ema_decay: float = 0.99):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size 
        self.num_quantizers = num_quantizers
        self.commitment_weight = commitment_weight
        
        # Create multiple quantizer layers
        self.quantizers = nn.ModuleList([
            VectorQuantizer(input_dim, codebook_size, commitment_weight, ema_decay)
            for _ in range(num_quantizers)
        ])
        
    def forward(self, x):
        """
        Forward pass through residual quantization layers.
        
        Args:
            x: Input tensor [batch, channels, time] or [batch, time, channels]
            
        Returns:
            quantized: Quantized representation
            codes: List of quantization codes for each layer
            losses: Dictionary of quantization losses
        """
        # Handle different input formats
        if x.dim() == 3 and x.shape[1] != self.input_dim:
            x = x.transpose(1, 2)  # [batch, time, channels] -> [batch, channels, time]
        
        residual = x
        quantized_layers = []
        codes = []
        total_loss = 0
        
        for i, quantizer in enumerate(self.quantizers):
            quantized, code, loss = quantizer(residual)
            quantized_layers.append(quantized)
            codes.append(code)
            total_loss += loss
            
            # Update residual for next layer
            residual = residual - quantized.detach()
        
        # Sum all quantized layers
        final_quantized = sum(quantized_layers)
        
        losses = {
            'vq_loss': total_loss,
            'num_layers': len(quantized_layers)
        }
        
        return final_quantized, codes, losses
    
    def encode(self, x):
        """Encode input to discrete codes."""
        _, codes, _ = self.forward(x)
        return codes
    
    def decode(self, codes):
        """Decode from discrete codes."""
        # Fixed: Use proper embedding reconstruction instead of codes[0] shape
        batch_size = codes[0].shape[0]
        time_steps = codes[0].shape[1]
        
        # Initialize with proper dimensions based on codebook
        quantized = torch.zeros(batch_size, self.input_dim, time_steps, 
                              dtype=torch.float, device=codes[0].device)
        
        for i, code in enumerate(codes):
            layer_quantized = self.quantizers[i].decode(code)
            quantized += layer_quantized
            
        return quantized


class VectorQuantizer(nn.Module):
    """Single vector quantizer layer with EMA updates."""
    
    def __init__(self, 
                 input_dim: int,
                 codebook_size: int, 
                 commitment_weight: float = 0.25,
                 ema_decay: float = 0.99):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay
        
        # Initialize codebook
        self.register_buffer('codebook', torch.randn(codebook_size, input_dim))
        self.register_buffer('ema_count', torch.zeros(codebook_size))
        self.register_buffer('ema_weight', self.codebook.clone())
        
    def forward(self, x):
        """Vector quantization forward pass."""
        # Flatten input for quantization
        input_shape = x.shape
        flat_input = x.view(-1, self.input_dim)
        
        # Calculate distances to codebook entries
        distances = torch.cdist(flat_input, self.codebook)
        
        # Find closest codebook entries
        codes = torch.argmin(distances, dim=1)
        quantized = F.embedding(codes, self.codebook)
        
        # Calculate losses
        e_latent_loss = F.mse_loss(quantized.detach(), flat_input)
        q_latent_loss = F.mse_loss(quantized, flat_input.detach())
        loss = q_latent_loss + self.commitment_weight * e_latent_loss
        
        # Straight-through estimator
        quantized = flat_input + (quantized - flat_input).detach()
        
        # Reshape back to input shape
        quantized = quantized.view(input_shape)
        # Fix: codes should match the spatial dimensions (batch, time), not include feature dim
        batch_size = input_shape[0]
        time_steps = input_shape[2] if len(input_shape) == 3 else flat_input.shape[0] // batch_size
        codes = codes.view(batch_size, time_steps)
        
        # EMA update during training
        if self.training:
            self._update_ema(flat_input, codes.view(-1))
        
        return quantized, codes, loss
    
    def decode(self, codes):
        """Decode codes to continuous representation."""
        # Handle different input shapes properly
        if codes.dim() == 2:  # [batch, time]
            batch_size, time_steps = codes.shape
            flat_codes = codes.view(-1)
            quantized_flat = F.embedding(flat_codes, self.codebook)
            return quantized_flat.view(batch_size, time_steps, self.input_dim).transpose(1, 2)
        else:
            return F.embedding(codes, self.codebook)
    
    def _update_ema(self, flat_input, codes):
        """Update codebook using exponential moving average."""
        with torch.no_grad():
            # Update counts
            codes_onehot = F.one_hot(codes, self.codebook_size).float()
            codes_count = codes_onehot.sum(dim=0)
            
            self.ema_count.mul_(self.ema_decay).add_(codes_count, alpha=1 - self.ema_decay)
            
            # Update weights
            codes_sum = torch.matmul(codes_onehot.t(), flat_input)
            self.ema_weight.mul_(self.ema_decay).add_(codes_sum, alpha=1 - self.ema_decay)
            
            # Update codebook
            n = self.ema_count.sum()
            weights = (self.ema_count + 1e-7) / (n + self.codebook_size * 1e-7)
            self.codebook.copy_(self.ema_weight / weights.unsqueeze(1))


class MelResidualEncoder(nn.Module):
    """
    Mel-scale residual encoder inspired by MuQ paper.
    Optimized for music-specific frequency representations.
    """
    def __init__(self,
                 n_mels: int = 128,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 target_dim: int = 512,
                 num_layers: int = 6):
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Mel spectrogram transform - will be updated dynamically
        self.mel_transform = T.MelSpectrogram(
            sample_rate=22050,  # Will be updated dynamically
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True
        )
        
        # Convolutional encoder stack
        self.encoder = nn.Sequential()
        in_channels = 1  # Start with 1 channel (mel spectrogram)
        
        for i in range(num_layers):
            out_channels = min(target_dim // (2 ** (num_layers - i - 1)), target_dim)
            
            self.encoder.add_module(f'conv_{i}', nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=2 if i < num_layers - 2 else 1,
                padding=1
            ))
            self.encoder.add_module(f'norm_{i}', nn.GroupNorm(8, out_channels))
            self.encoder.add_module(f'act_{i}', nn.GELU())
            
            in_channels = out_channels
        
        # Final projection to target dimension
        self.proj = nn.Conv2d(in_channels, target_dim, kernel_size=1)
        
    def forward(self, waveform, sample_rate=22050):
        """Encode waveform to mel-based representation."""
        self.mel_transform = self.mel_transform.to(waveform.device)
        
        # Update sample rate if needed
        if hasattr(self.mel_transform, 'sample_rate'):
            self.mel_transform.sample_rate = sample_rate
        
        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Add batch dimension if needed
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1)  # [batch, 1, mels, time]
        
        # Encode through convolutional layers
        encoded = self.encoder(mel_spec)
        encoded = self.proj(encoded)
        
        # Global average pooling over frequency axis
        encoded = encoded.mean(dim=2)  # [batch, channels, time]
        
        return encoded


class SemanticAudioEncoder(nn.Module):
    """
    Semantic audio encoder using pre-trained models like Wav2Vec2.
    Extracts high-level musical concepts and structures.
    """
    def __init__(self, model_name: str = "facebook/wav2vec2-base", target_dim: int = 512):
        super().__init__()
        self.model_name = model_name
        self.target_dim = target_dim
        self.fallback_proj = None  # Cache for fallback projection
        
        if HAS_TRANSFORMERS:
            try:
                self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
                self.processor = Wav2Vec2Processor.from_pretrained(model_name)
                
                # Projection layer to target dimension
                wav2vec_dim = self.wav2vec2.config.hidden_size
                self.projection = nn.Linear(wav2vec_dim, target_dim)
                
                # Freeze wav2vec2 parameters 
                for param in self.wav2vec2.parameters():
                    param.requires_grad = False
                    
                self.available = True
            except Exception as e:
                print(f"Warning: Could not load Wav2Vec2 model: {e}")
                self.available = False
        else:
            self.available = False
    
    def forward(self, waveform, sample_rate=16000):
        """Extract semantic features from waveform."""
        if not self.available:
            # Fallback to simple spectral features
            return self._spectral_fallback(waveform, sample_rate)
        
        # Simple device matching - just move models to input device
        self.wav2vec2 = self.wav2vec2.to(waveform.device)
        self.projection = self.projection.to(waveform.device)
        
        # Resample to 16kHz if needed (Wav2Vec2 requirement)
        if sample_rate != 16000:
            resampler = T.Resample(sample_rate, 16000).to(waveform.device)
            waveform = resampler(waveform)
        
        # Process through Wav2Vec2
        with torch.no_grad():
            features = self.wav2vec2(waveform.squeeze(0) if waveform.dim() > 1 else waveform)
            hidden_states = features.last_hidden_state
        
        # Project to target dimension
        projected = self.projection(hidden_states)
        
        # Transpose to [batch, channels, time] format
        return projected.transpose(1, 2)
    
    def _spectral_fallback(self, waveform, sample_rate):
        """Fixed fallback spectral features with persistent projection layer."""
        # Ensure waveform is 1D for processing
        if waveform.dim() > 1:
            audio_1d = waveform.squeeze()
        else:
            audio_1d = waveform
        
        # STFT with scipy-like processing
        n_fft, hop = 2048, 512
        win = torch.hann_window(n_fft, device=waveform.device)
        
        # Calculate number of frames
        if len(audio_1d) >= n_fft:
            num_frames = 1 + (len(audio_1d) - n_fft) // hop
        else:
            num_frames = 1
            
        # Compute STFT frames
        stft_frames = []
        for i in range(num_frames):
            start = i * hop
            end = start + n_fft
            frame = audio_1d[start:end] if end <= len(audio_1d) else torch.cat([
                audio_1d[start:], torch.zeros(end - len(audio_1d), device=waveform.device)
            ])
            windowed = frame * win
            stft_frame = torch.fft.rfft(windowed)
            stft_frames.append(stft_frame)
        
        stft_result = torch.stack(stft_frames, dim=1)  # [freq, time]
        magnitude = torch.abs(stft_result) + 1e-12
        
        # Frequency axis for centroid calculation
        freqs = torch.fft.rfftfreq(n_fft, 1.0/sample_rate, device=waveform.device)
        freqs = freqs.unsqueeze(1)  # [freq, 1]
        
        # Spectral centroid and bandwidth
        total_mag = magnitude.sum(dim=0) + 1e-8  # [time]
        centroid = (magnitude * freqs).sum(dim=0) / total_mag  # [time]
        
        # Spectral bandwidth
        freq_diff_sq = (freqs - centroid.unsqueeze(0)) ** 2
        bandwidth = torch.sqrt((magnitude * freq_diff_sq).sum(dim=0) / total_mag)  # [time]
        
        # Stack features [2, time]
        features = torch.stack([centroid, bandwidth], dim=0)  # [2, time]
        
        # Create persistent projection layer (fixed memory leak)
        if self.fallback_proj is None or self.fallback_proj.weight.device != waveform.device:
            self.fallback_proj = nn.Linear(2, self.target_dim).to(waveform.device)
        
        # Project to target dimension
        projected = self.fallback_proj(features.transpose(0, 1)).transpose(0, 1)  # [target_dim, time]
        
        return projected.unsqueeze(0)  # Add batch dimension [1, target_dim, time]


# ============================================================================
# Multi-Scale Temporal Modeling
# ============================================================================

class MultiScaleTemporalEncoder(nn.Module):
    """
    Multi-scale temporal encoder for capturing rhythm, harmony, and structure
    at different time scales simultaneously.
    """
    def __init__(self,
                 input_dim: int = 512,
                 scales: List[int] = [1, 2, 4, 8, 16],  # Frame scales
                 hidden_dim: int = 256):
        super().__init__()
        self.scales = scales
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Scale-specific encoders
        self.scale_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=scale*2+1, 
                         stride=scale, padding=scale),
                nn.GroupNorm(8, hidden_dim),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.GELU()
            ) for scale in scales
        ])
        
        # Fusion layer
        self.fusion = nn.Conv1d(hidden_dim * len(scales), input_dim, kernel_size=1)
        
    def forward(self, x):
        """Multi-scale temporal processing."""
        scale_features = []
        target_length = x.shape[-1]
        
        for scale_idx, encoder in enumerate(self.scale_encoders):
            features = encoder(x)
            
            # Upsample to match target length
            if features.shape[-1] != target_length:
                features = F.interpolate(features, size=target_length, mode='linear', align_corners=False)
            
            scale_features.append(features)
        
        # Concatenate and fuse
        combined = torch.cat(scale_features, dim=1)
        fused = self.fusion(combined)
        
        return fused


# ============================================================================
# NDJSON Streaming Protocol (LLM-friendly LAM v0.1)
# ============================================================================

class NDJSONStreamer:
    """Enhanced NDJSON streaming protocol for LLM-friendly audio tokenization."""
    
    def __init__(self, sample_rate: int, hop_length: int, model_id: str = "tims-ears-v1.0", 
                 codebook_size: int = 1024, num_semantic_layers: int = 4, num_acoustic_layers: int = 4,
                 rle_mode: bool = False, per_layer_encoding: Optional[Dict[str, str]] = None,
                 keyframe_interval_seconds: float = 5.0, audio_sha256: Optional[str] = None):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.hop_ms = (hop_length / sample_rate) * 1000.0
        self.frames_per_second = sample_rate / hop_length
        self.frame_duration_ms = 1000.0 / self.frames_per_second
        self.model_id = model_id
        self.codebook_size = codebook_size
        self.num_semantic_layers = num_semantic_layers
        self.num_acoustic_layers = num_acoustic_layers
        self.rle_mode = rle_mode
        self.per_layer_encoding = per_layer_encoding or {}
        self.keyframe_interval_seconds = keyframe_interval_seconds
        self.audio_sha256 = audio_sha256
        
        # RLE state for duration aggregation
        self.buffered_event = None
        self.last_frame_index = -1
        
    def create_header(self, duration_seconds: float = None, metadata: Dict = None, 
                     include_legend: bool = True) -> str:
        """Create enhanced NDJSON header with full format specification."""
        layers = []
        
        # Semantic layers with encoding preferences
        for i in range(self.num_semantic_layers):
            layer_name = f"S{i}"
            encoding = self.per_layer_encoding.get(layer_name, "rle" if self.rle_mode else "dense")
            layers.append({
                "name": layer_name,
                "type": "semantic", 
                "vocab": self.codebook_size,
                "encoding": encoding
            })
        
        # Acoustic layers with encoding preferences
        for i in range(self.num_acoustic_layers):
            layer_name = f"A{i}"
            encoding = self.per_layer_encoding.get(layer_name, "dense")  # A-layers default to dense
            layers.append({
                "name": layer_name,
                "type": "acoustic",
                "vocab": self.codebook_size,
                "encoding": encoding
            })
        
        header = {
            "event": "header",
            "format_version": "1.0",
            "schema": "lam_audio_tokens",
            "model_id": self.model_id,
            "codebook_id": f"{self.model_id}-cb-{self.codebook_size}",
            "sr": self.sample_rate,
            "hop": self.hop_length,
            "hop_ms": round(self.hop_ms, 3),
            "frames_per_second": round(self.frames_per_second, 3),
            "encoding_mode": "rle" if self.rle_mode else "dense",
            "time_units": "ms",
            "start_ts": 0.0,
            "layers": layers
        }
        
        if include_legend:
            header["legend"] = "S* encodes slower, scene/gesture level; A* encodes timbre/texture/transient detail; S0 < S1 < S2 < S3 in timescale"
            
        if duration_seconds is not None:
            header["duration_seconds"] = round(duration_seconds, 3)
            
        if self.audio_sha256:
            header["audio_sha256"] = self.audio_sha256
            
        if metadata:
            header["metadata"] = metadata
            
        return json.dumps(header, separators=(',', ':'))
    
    def _should_use_rle_for_layer(self, layer_name: str) -> bool:
        """Determine if a specific layer should use RLE encoding."""
        return self.per_layer_encoding.get(layer_name, 
            "rle" if (self.rle_mode and layer_name.startswith('S')) else "dense"
        ) == "rle"
    
    def _flush_buffered_event(self) -> Optional[str]:
        """Flush any buffered RLE event and return the JSON string."""
        if self.buffered_event is None:
            return None
            
        event_json = json.dumps(self.buffered_event, separators=(',', ':'))
        self.buffered_event = None
        return event_json
    
    def create_frame(self, frame_index: int, time_ms: float, semantic_tokens: List[int], 
                    acoustic_tokens: List[int], changed_layers: List[str] = None,
                    is_keyframe: bool = False, aux_data: Dict = None) -> Optional[str]:
        """Create NDJSON frame/event line with RLE aggregation and keyframe support."""
        
        # Defensive check for layer count consistency
        expected_sem_layers = self.num_semantic_layers
        expected_acc_layers = self.num_acoustic_layers
        
        if len(semantic_tokens) != expected_sem_layers:
            print(f"Warning: Expected {expected_sem_layers} semantic tokens, got {len(semantic_tokens)}")
            # Pad or truncate to expected size
            if len(semantic_tokens) < expected_sem_layers:
                semantic_tokens.extend([0] * (expected_sem_layers - len(semantic_tokens)))
            else:
                semantic_tokens = semantic_tokens[:expected_sem_layers]
                
        if len(acoustic_tokens) != expected_acc_layers:
            print(f"Warning: Expected {expected_acc_layers} acoustic tokens, got {len(acoustic_tokens)}")
            # Pad or truncate to expected size
            if len(acoustic_tokens) < expected_acc_layers:
                acoustic_tokens.extend([0] * (expected_acc_layers - len(acoustic_tokens)))
            else:
                acoustic_tokens = acoustic_tokens[:expected_acc_layers]
        
        # Force keyframe or dense mode
        if is_keyframe or not self.rle_mode:
            # Flush any buffered RLE event first
            flushed = self._flush_buffered_event()
            
            # Create dense frame event
            event = {
                "event": "frame",
                "fi": frame_index,
                "ts": round(time_ms, 3),
                "dur": round(self.frame_duration_ms, 3),
                "S": [int(token) for token in semantic_tokens],
                "A": [int(token) for token in acoustic_tokens]
            }
            
            if is_keyframe:
                event["is_keyframe"] = True
                
            if aux_data:
                event["aux"] = aux_data
                
            result = json.dumps(event, separators=(',', ':'))
            
            # Return both flushed and current event
            if flushed:
                return flushed + '\n' + result
            return result
        
        # RLE mode with duration aggregation
        if changed_layers:
            # Check if we should buffer or extend previous event
            if self.buffered_event is not None:
                # Extend duration of buffered event
                frames_elapsed = frame_index - self.last_frame_index
                additional_duration = frames_elapsed * self.frame_duration_ms
                self.buffered_event["dur"] += additional_duration
                
                # Flush the buffered event
                flushed = self._flush_buffered_event()
            else:
                flushed = None
            
            # Create new RLE tokens event
            event = {
                "event": "tokens",
                "fi": frame_index,
                "ts": round(time_ms, 3),
                "dur": round(self.frame_duration_ms, 3)  # Initial duration
            }
            
            # Add only changed layers that should use RLE
            for layer_name in changed_layers:
                if layer_name.startswith('S'):
                    layer_idx = int(layer_name[1:])
                    if layer_idx < len(semantic_tokens) and self._should_use_rle_for_layer(layer_name):
                        event[layer_name] = int(semantic_tokens[layer_idx])
                elif layer_name.startswith('A'):
                    layer_idx = int(layer_name[1:])
                    if layer_idx < len(acoustic_tokens) and self._should_use_rle_for_layer(layer_name):
                        event[layer_name] = int(acoustic_tokens[layer_idx])
            
            # Also include any dense-mode layers in full
            dense_layers_s = [int(token) for i, token in enumerate(semantic_tokens) 
                            if not self._should_use_rle_for_layer(f"S{i}")]
            dense_layers_a = [int(token) for i, token in enumerate(acoustic_tokens)
                            if not self._should_use_rle_for_layer(f"A{i}")]
            
            if dense_layers_s:
                event["S_dense"] = dense_layers_s
            if dense_layers_a:
                event["A_dense"] = dense_layers_a
                
            if aux_data:
                event["aux"] = aux_data
            
            # Buffer this event for potential duration extension
            self.buffered_event = event
            self.last_frame_index = frame_index
            
            # Return flushed event if any
            return flushed
        else:
            # No changes - just extend buffered event duration if exists
            if self.buffered_event is not None:
                frames_elapsed = frame_index - self.last_frame_index
                additional_duration = frames_elapsed * self.frame_duration_ms
                self.buffered_event["dur"] += additional_duration
                self.last_frame_index = frame_index
            
            return None  # No event to emit
    
    def create_end_marker(self, stats: Dict = None) -> str:
        """Create end-of-stream marker, flushing any buffered events."""
        lines = []
        
        # Flush any remaining buffered event
        flushed = self._flush_buffered_event()
        if flushed:
            lines.append(flushed)
        
        # Create end event
        end_event = {"event": "end"}
        if stats:
            end_event["stats"] = stats
        lines.append(json.dumps(end_event, separators=(',', ':')))
        
        return '\n'.join(lines) if len(lines) > 1 else lines[0]


# ============================================================================
# Token Budget and Throughput Meters
# ============================================================================

@dataclass
class TokenBudgetMetrics:
    """Token budget and throughput tracking metrics."""
    total_tokens: int = 0
    semantic_tokens: int = 0
    acoustic_tokens: int = 0
    tokens_per_second: float = 0.0
    frames_per_second: float = 0.0
    compression_ratio: float = 0.0
    processing_time: float = 0.0


class TokenBudgetMeter:
    """Track token budget and throughput for LLM consumption with accurate timebase calculation."""
    
    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frames_per_second = sample_rate / hop_length  # Accurate FPS calculation
        self.reset()
    
    def reset(self):
        """Reset all counters."""
        self.start_time = time.time()
        self.total_samples = 0
        self.total_frames = 0
        self.semantic_tokens = 0
        self.acoustic_tokens = 0
    
    def update(self, samples: int, frames: int, sem_tokens: int, acc_tokens: int):
        """Update counters with new data."""
        self.total_samples += samples
        self.total_frames += frames
        self.semantic_tokens += sem_tokens
        self.acoustic_tokens += acc_tokens
    
    def get_metrics(self) -> TokenBudgetMetrics:
        """Get current metrics with accurate calculations."""
        elapsed = time.time() - self.start_time
        total_tokens = self.semantic_tokens + self.acoustic_tokens
        
        # Calculate accurate frames_per_second based on actual frame count and audio duration
        audio_duration = self.total_samples / self.sample_rate if self.sample_rate > 0 else elapsed
        actual_fps = self.total_frames / max(audio_duration, 1e-6)
        
        return TokenBudgetMetrics(
            total_tokens=total_tokens,
            semantic_tokens=self.semantic_tokens,
            acoustic_tokens=self.acoustic_tokens,
            tokens_per_second=total_tokens / max(elapsed, 1e-6),
            frames_per_second=actual_fps,  # Use actual FPS from data
            compression_ratio=self.total_samples / max(total_tokens, 1),
            processing_time=elapsed
        )


# ============================================================================
# Main Neural Audio Tokenizer
# ============================================================================

class NeuralAudioTokenizer(nn.Module):
    """
    Main neural audio tokenizer implementing hybrid semantic + acoustic approach
    based on AudioLM and MuQ research.
    """
    def __init__(self,
                 sample_rate: int = 22050,
                 semantic_dim: int = 512,
                 acoustic_dim: int = 512, 
                 codebook_size: int = 1024,
                 num_quantizers: int = 8,
                 n_mels: int = 128,
                 hop_length: int = 512):
        super().__init__()
        self.sample_rate = sample_rate
        self.semantic_dim = semantic_dim
        self.acoustic_dim = acoustic_dim
        self.hop_length = hop_length
        
        # Semantic encoder (high-level musical concepts)
        self.semantic_encoder = SemanticAudioEncoder(target_dim=semantic_dim)
        
        # Acoustic encoder (fine-grained audio details)
        self.acoustic_encoder = MelResidualEncoder(
            n_mels=n_mels,
            hop_length=hop_length,
            target_dim=acoustic_dim
        )
        
        # Multi-scale temporal modeling
        self.temporal_semantic = MultiScaleTemporalEncoder(semantic_dim)
        self.temporal_acoustic = MultiScaleTemporalEncoder(acoustic_dim)
        
        # Quantizers
        self.semantic_quantizer = ResidualVectorQuantizer(
            semantic_dim, codebook_size, num_quantizers//2
        )
        self.acoustic_quantizer = ResidualVectorQuantizer(
            acoustic_dim, codebook_size, num_quantizers//2
        )
        
        # Decoder for reconstruction
        self.decoder = self._build_decoder()
        
    def _build_decoder(self):
        """Build decoder for audio reconstruction."""
        return nn.Sequential(
            nn.Conv1d(self.semantic_dim + self.acoustic_dim, 512, kernel_size=3, padding=1),
            nn.GroupNorm(8, 512),
            nn.GELU(),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv1d(128, 1, kernel_size=3, padding=1)  # Output single channel
        )
    
    def forward(self, waveform):
        """
        Forward pass through complete tokenization pipeline.
        
        Returns:
            semantic_codes: High-level musical structure tokens
            acoustic_codes: Fine-grained audio detail tokens  
            losses: Training losses
            reconstructed: Reconstructed audio (if decoder enabled)
        """
        batch_size = waveform.shape[0]
        
        # Extract semantic and acoustic features
        semantic_features = self.semantic_encoder(waveform, self.sample_rate)
        acoustic_features = self.acoustic_encoder(waveform, self.sample_rate)
        
        # Multi-scale temporal modeling
        semantic_features = self.temporal_semantic(semantic_features)
        acoustic_features = self.temporal_acoustic(acoustic_features)
        
        # Explicit time-base alignment before quantization
        T_sem = semantic_features.shape[-1]
        T_acc = acoustic_features.shape[-1] 
        T_target = min(T_sem, T_acc)
        
        if T_sem != T_target:
            semantic_features = F.interpolate(
                semantic_features, size=T_target, mode='linear', align_corners=False
            )
        if T_acc != T_target:
            acoustic_features = F.interpolate(
                acoustic_features, size=T_target, mode='linear', align_corners=False
            )
        
        # Quantization
        semantic_quantized, semantic_codes, semantic_losses = self.semantic_quantizer(semantic_features)
        acoustic_quantized, acoustic_codes, acoustic_losses = self.acoustic_quantizer(acoustic_features)
        
        # Combine losses
        total_losses = {
            'semantic_vq_loss': semantic_losses['vq_loss'],
            'acoustic_vq_loss': acoustic_losses['vq_loss'],
            'total_vq_loss': semantic_losses['vq_loss'] + acoustic_losses['vq_loss']
        }
        
        # Optional reconstruction
        reconstructed = None
        if hasattr(self, 'decoder') and self.decoder is not None:
            # Combine semantic and acoustic features for decoding  
            combined_features = torch.cat([semantic_quantized, acoustic_quantized], dim=1)
            
            # Decode to frame domain
            decoded_frames = self.decoder(combined_features)  # [batch, 1, T_target]
            
            # Calculate target length from actual hop size and frame count
            target_waveform_length = T_target * self.hop_length
            target_waveform_length = min(target_waveform_length, waveform.shape[-1])
            
            # Upsample to target waveform length
            reconstructed = F.interpolate(
                decoded_frames, size=target_waveform_length,
                mode='linear', align_corners=False
            )
            
            # Ensure reconstruction matches input format for loss calculation
            waveform_for_loss = waveform
            if waveform.dim() == 2:  # [batch, samples]
                waveform_for_loss = waveform.unsqueeze(1)  # [batch, 1, samples]
            
            # Align lengths for loss calculation
            min_len = min(waveform_for_loss.shape[-1], reconstructed.shape[-1])
            recon_loss = F.mse_loss(
                reconstructed[..., :min_len], 
                waveform_for_loss[..., :min_len]
            )
            total_losses['reconstruction_loss'] = recon_loss
        
        return {
            'semantic_codes': semantic_codes,
            'acoustic_codes': acoustic_codes,
            'losses': total_losses,
            'reconstructed': reconstructed,
            'semantic_features': semantic_features,
            'acoustic_features': acoustic_features,
            'num_frames': T_target
        }
    
    def encode(self, waveform):
        """Encode audio to discrete tokens."""
        with torch.no_grad():
            result = self.forward(waveform)
            return result['semantic_codes'], result['acoustic_codes']
    
    def decode_tokens(self, semantic_codes, acoustic_codes):
        """Decode tokens back to audio (if decoder available)."""
        if not hasattr(self, 'decoder') or self.decoder is None:
            raise NotImplementedError("Decoder not available")
        
        with torch.no_grad():
            # Decode from quantizers
            semantic_features = self.semantic_quantizer.decode(semantic_codes)
            acoustic_features = self.acoustic_quantizer.decode(acoustic_codes)
            
            # Combine and decode
            combined_features = torch.cat([semantic_features, acoustic_features], dim=1)
            reconstructed = self.decoder(combined_features)
            
            return reconstructed


# ============================================================================
# Evaluation and Analysis Metrics
# ============================================================================

@dataclass
class TokenizationMetrics:
    """Comprehensive metrics for evaluating tokenization quality."""
    # Basic statistics
    num_semantic_tokens: int
    num_acoustic_tokens: int
    compression_ratio: float
    token_diversity: float
    
    # Reconstruction metrics
    mse_loss: float
    spectral_loss: float
    perceptual_loss: float
    
    # Information theory metrics
    semantic_entropy: float
    acoustic_entropy: float
    mutual_information: float
    
    # Music-specific metrics
    pitch_accuracy: float
    rhythm_accuracy: float
    timbral_similarity: float
    
    # Efficiency metrics
    encoding_time: float
    decoding_time: float
    memory_usage: float
    
    # Token budget metrics
    tokens_per_second: float = 0.0
    frames_per_second: float = 0.0


class TokenizationEvaluator:
    """Scientific evaluation of tokenization approaches."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        
    def evaluate_tokenization(self, 
                            original_audio: np.ndarray,
                            tokenizer: NeuralAudioTokenizer,
                            reconstruction: Optional[np.ndarray] = None) -> TokenizationMetrics:
        """Comprehensive evaluation of tokenization quality."""
        
        # Convert to torch tensor and move to same device as tokenizer
        if isinstance(original_audio, np.ndarray):
            audio_tensor = torch.from_numpy(original_audio).float().unsqueeze(0)
        else:
            audio_tensor = original_audio
        
        # Ensure audio tensor is on same device as tokenizer
        tokenizer_device = next(tokenizer.parameters()).device
        audio_tensor = audio_tensor.to(tokenizer_device)
        
        # Time encoding
        start_time = time.time()
        
        with torch.no_grad():
            result = tokenizer(audio_tensor)
            semantic_codes = result['semantic_codes']
            acoustic_codes = result['acoustic_codes']
            reconstructed = result['reconstructed']
        
        encoding_time = time.time() - start_time
        
        # Basic statistics
        num_semantic = sum(codes.numel() for codes in semantic_codes)
        num_acoustic = sum(codes.numel() for codes in acoustic_codes)
        total_tokens = num_semantic + num_acoustic
        
        # Compression ratio (original samples vs tokens)
        compression_ratio = len(original_audio) / total_tokens if total_tokens > 0 else 0
        
        # Token diversity (unique tokens / total tokens)
        all_semantic = torch.cat([codes.flatten().long().cpu() for codes in semantic_codes]) if semantic_codes else torch.tensor([])
        all_acoustic = torch.cat([codes.flatten().long().cpu() for codes in acoustic_codes]) if acoustic_codes else torch.tensor([])
        
        semantic_diversity = len(torch.unique(all_semantic)) / len(all_semantic) if len(all_semantic) > 0 else 0
        acoustic_diversity = len(torch.unique(all_acoustic)) / len(all_acoustic) if len(all_acoustic) > 0 else 0
        token_diversity = (semantic_diversity + acoustic_diversity) / 2
        
        # Reconstruction metrics
        mse_loss = 0.0
        spectral_loss = 0.0
        perceptual_loss = 0.0
        
        if reconstructed is not None:
            recon_audio = reconstructed.squeeze().cpu().numpy()
            
            # Ensure same length
            min_len = min(len(original_audio), len(recon_audio))
            orig_aligned = original_audio[:min_len]
            recon_aligned = recon_audio[:min_len]
            
            # MSE loss
            mse_loss = float(np.mean((orig_aligned - recon_aligned) ** 2))
            
            # Spectral loss
            orig_spec = np.abs(librosa.stft(orig_aligned))
            recon_spec = np.abs(librosa.stft(recon_aligned))
            spectral_loss = float(np.mean((orig_spec - recon_spec) ** 2))
            
            # Perceptual loss (MFCC-based)
            orig_mfcc = librosa.feature.mfcc(y=orig_aligned, sr=self.sample_rate)
            recon_mfcc = librosa.feature.mfcc(y=recon_aligned, sr=self.sample_rate)
            perceptual_loss = float(np.mean((orig_mfcc - recon_mfcc) ** 2))
        
        # Information theory metrics
        semantic_entropy = self._calculate_entropy(all_semantic) if len(all_semantic) > 0 else 0
        acoustic_entropy = self._calculate_entropy(all_acoustic) if len(all_acoustic) > 0 else 0
        mutual_information = self._calculate_mutual_information(all_semantic, all_acoustic)
        
        # Music-specific metrics
        pitch_accuracy = self._evaluate_pitch_preservation(original_audio, reconstructed)
        rhythm_accuracy = self._evaluate_rhythm_preservation(original_audio, reconstructed) 
        timbral_similarity = self._evaluate_timbral_similarity(original_audio, reconstructed)
        
        # Throughput metrics
        frames_per_second = result.get('num_frames', 0) / max(encoding_time, 1e-6)
        tokens_per_second = total_tokens / max(encoding_time, 1e-6)
        
        return TokenizationMetrics(
            num_semantic_tokens=num_semantic,
            num_acoustic_tokens=num_acoustic,
            compression_ratio=compression_ratio,
            token_diversity=token_diversity,
            mse_loss=mse_loss,
            spectral_loss=spectral_loss,
            perceptual_loss=perceptual_loss,
            semantic_entropy=semantic_entropy,
            acoustic_entropy=acoustic_entropy,
            mutual_information=mutual_information,
            pitch_accuracy=pitch_accuracy,
            rhythm_accuracy=rhythm_accuracy,
            timbral_similarity=timbral_similarity,
            encoding_time=encoding_time,
            decoding_time=0.0,  # TODO: Implement
            memory_usage=0.0,   # TODO: Implement
            tokens_per_second=tokens_per_second,
            frames_per_second=frames_per_second
        )
    
    def _calculate_entropy(self, tokens: torch.Tensor) -> float:
        """Calculate entropy of token distribution."""
        if len(tokens) == 0:
            return 0.0
        
        unique, counts = torch.unique(tokens, return_counts=True)
        probabilities = counts.float() / len(tokens)
        return float(entropy(probabilities.cpu().numpy()))
    
    def _calculate_mutual_information(self, tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> float:
        """Fixed mutual information calculation using numpy histogram2d."""
        if len(tokens_a) == 0 or len(tokens_b) == 0:
            return 0.0
        
        # Convert to numpy and align lengths
        a_np = tokens_a.cpu().numpy().astype(np.int64).ravel()
        b_np = tokens_b.cpu().numpy().astype(np.int64).ravel()
        min_len = min(len(a_np), len(b_np))
        
        if min_len == 0:
            return 0.0
            
        a_aligned = a_np[:min_len]
        b_aligned = b_np[:min_len]
        
        # Use numpy's histogram2d for proper joint distribution
        try:
            hist_2d, x_edges, y_edges = np.histogram2d(a_aligned, b_aligned, bins=min(64, max(len(np.unique(a_aligned)), len(np.unique(b_aligned)))))
            
            # Convert to probabilities
            pxy = hist_2d / (hist_2d.sum() + 1e-12)
            px = pxy.sum(axis=1, keepdims=True)
            py = pxy.sum(axis=0, keepdims=True) 
            
            # Calculate MI
            nonzero = pxy > 1e-12
            mi_val = np.sum(pxy[nonzero] * np.log2(pxy[nonzero] / (px[nonzero.any(axis=1)][:, None] * py[None, nonzero.any(axis=0)] + 1e-12)))
            
            return float(mi_val) if not np.isnan(mi_val) else 0.0
        except:
            return 0.0
    
    def _evaluate_pitch_preservation(self, original: np.ndarray, reconstructed: Optional[torch.Tensor]) -> float:
        """Evaluate how well pitch information is preserved."""
        if reconstructed is None:
            return 0.0
        
        try:
            recon_np = reconstructed.squeeze().cpu().numpy()
            
            # Extract pitch tracks
            orig_pitches = librosa.piptrack(y=original, sr=self.sample_rate)[0]
            recon_pitches = librosa.piptrack(y=recon_np, sr=self.sample_rate)[0]
            
            # Compare dominant pitches per frame
            orig_dominant = np.array([frame[frame.argmax()] for frame in orig_pitches.T])
            recon_dominant = np.array([frame[frame.argmax()] for frame in recon_pitches.T])
            
            min_len = min(len(orig_dominant), len(recon_dominant))
            if min_len < 2:  # Need at least 2 points for correlation
                return 0.0
            
            # Align arrays
            orig_aligned = orig_dominant[:min_len]
            recon_aligned = recon_dominant[:min_len]
            
            # Check for valid data and variation before calling corrcoef
            if (np.std(orig_aligned) == 0 or np.std(recon_aligned) == 0 or
                np.any(np.isnan(orig_aligned)) or np.any(np.isnan(recon_aligned)) or
                np.any(np.isinf(orig_aligned)) or np.any(np.isinf(recon_aligned))):
                return 0.0
            
            # Correlation coefficient
            correlation = np.corrcoef(orig_aligned, recon_aligned)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _evaluate_rhythm_preservation(self, original: np.ndarray, reconstructed: Optional[torch.Tensor]) -> float:
        """Evaluate rhythm/onset preservation."""
        if reconstructed is None:
            return 0.0
        
        try:
            recon_np = reconstructed.squeeze().cpu().numpy()
            
            # Extract onset patterns
            orig_onsets = librosa.onset.onset_detect(y=original, sr=self.sample_rate, units='time')
            recon_onsets = librosa.onset.onset_detect(y=recon_np, sr=self.sample_rate, units='time')
            
            if len(orig_onsets) < 2 or len(recon_onsets) < 2:  # Need at least 2 onsets for intervals
                return 0.0
            
            # Calculate onset timing similarity (simplified)
            orig_intervals = np.diff(orig_onsets)
            recon_intervals = np.diff(recon_onsets)
            
            min_len = min(len(orig_intervals), len(recon_intervals))
            if min_len < 2:  # Need at least 2 intervals for correlation
                return 0.0
            
            # Align arrays
            orig_aligned = orig_intervals[:min_len]
            recon_aligned = recon_intervals[:min_len]
            
            # Check for valid data and variation before calling corrcoef
            if (np.std(orig_aligned) == 0 or np.std(recon_aligned) == 0 or
                np.any(np.isnan(orig_aligned)) or np.any(np.isnan(recon_aligned)) or
                np.any(np.isinf(orig_aligned)) or np.any(np.isinf(recon_aligned))):
                return 0.0
            
            correlation = np.corrcoef(orig_aligned, recon_aligned)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _evaluate_timbral_similarity(self, original: np.ndarray, reconstructed: Optional[torch.Tensor]) -> float:
        """Evaluate timbral similarity using MFCC."""
        if reconstructed is None:
            return 0.0
        
        try:
            recon_np = reconstructed.squeeze().cpu().numpy()
            
            # Extract MFCC features
            orig_mfcc = librosa.feature.mfcc(y=original, sr=self.sample_rate, n_mfcc=13)
            recon_mfcc = librosa.feature.mfcc(y=recon_np, sr=self.sample_rate, n_mfcc=13)
            
            # Cosine similarity between average MFCC vectors
            orig_mean = np.mean(orig_mfcc, axis=1)
            recon_mean = np.mean(recon_mfcc, axis=1)
            
            similarity = np.dot(orig_mean, recon_mean) / (np.linalg.norm(orig_mean) * np.linalg.norm(recon_mean) + 1e-8)
            return float(similarity) if not np.isnan(similarity) else 0.0
        except:
            return 0.0

    def generate_visualizations(self, original_audio: np.ndarray, result: Dict, output_dir: str, base_name: str, sequential: bool = False):
        """Generate comprehensive visualizations and save to files."""
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not available, skipping visualizations")
            return {}
        
        viz_files = {}
        
        if sequential:
            return self._generate_visualizations_sequential(original_audio, result, output_dir, base_name)
        else:
            return self._generate_visualizations_parallel(original_audio, result, output_dir, base_name)
    
    def _generate_visualizations_sequential(self, original_audio: np.ndarray, result: Dict, output_dir: str, base_name: str):
        """Generate visualizations one at a time with memory cleanup."""
        viz_files = {}
        
        try:
            plt.style.use('default')
            
            # 1. Original vs Reconstructed Waveforms
            if result.get('reconstructed') is not None:
                print("  Generating waveform comparison...")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
                
                time_orig = np.linspace(0, len(original_audio) / self.sample_rate, len(original_audio))
                ax1.plot(time_orig, original_audio, 'b-', alpha=0.7, linewidth=0.5)
                ax1.set_title('Original Waveform')
                ax1.set_ylabel('Amplitude')
                ax1.grid(True, alpha=0.3)
                
                recon_audio = result['reconstructed'].squeeze().cpu().numpy()
                time_recon = np.linspace(0, len(recon_audio) / self.sample_rate, len(recon_audio))
                ax2.plot(time_recon, recon_audio, 'r-', alpha=0.7, linewidth=0.5)
                ax2.set_title('Reconstructed Waveform')
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Amplitude')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                waveform_file = Path(output_dir) / f"{base_name}_waveforms.png"
                plt.savefig(waveform_file, dpi=150, bbox_inches='tight')
                plt.close()
                viz_files['waveforms'] = str(waveform_file)
                
                # Clear memory
                del recon_audio, time_orig, time_recon
                gc.collect()
            
            # 2. Spectrograms comparison
            print("  Generating spectrograms...")
            
            # Original spectrogram
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(original_audio)), ref=np.max)
            img1 = librosa.display.specshow(D_orig, y_axis='hz', x_axis='time', 
                                          sr=self.sample_rate, ax=ax)
            ax.set_title('Original Spectrogram')
            plt.colorbar(img1, ax=ax, format='%+2.0f dB')
            plt.tight_layout()
            orig_spec_file = Path(output_dir) / f"{base_name}_original_spectrogram.png"
            plt.savefig(orig_spec_file, dpi=150, bbox_inches='tight')
            plt.close()
            viz_files['orig_spectrogram'] = str(orig_spec_file)
            del D_orig
            gc.collect()
            
            # Original mel spectrogram
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            mel_orig = librosa.feature.melspectrogram(y=original_audio, sr=self.sample_rate)
            mel_db_orig = librosa.power_to_db(mel_orig, ref=np.max)
            img2 = librosa.display.specshow(mel_db_orig, y_axis='mel', x_axis='time',
                                          sr=self.sample_rate, ax=ax)
            ax.set_title('Original Mel Spectrogram')
            plt.colorbar(img2, ax=ax, format='%+2.0f dB')
            plt.tight_layout()
            orig_mel_file = Path(output_dir) / f"{base_name}_original_mel_spectrogram.png"
            plt.savefig(orig_mel_file, dpi=150, bbox_inches='tight')
            plt.close()
            viz_files['orig_mel_spectrogram'] = str(orig_mel_file)
            del mel_orig, mel_db_orig
            gc.collect()
            
            if result.get('reconstructed') is not None:
                recon_audio = result['reconstructed'].squeeze().cpu().numpy()
                
                # Reconstructed spectrogram
                fig, ax = plt.subplots(1, 1, figsize=(8, 5))
                D_recon = librosa.amplitude_to_db(np.abs(librosa.stft(recon_audio)), ref=np.max)
                img3 = librosa.display.specshow(D_recon, y_axis='hz', x_axis='time',
                                              sr=self.sample_rate, ax=ax)
                ax.set_title('Reconstructed Spectrogram')
                plt.colorbar(img3, ax=ax, format='%+2.0f dB')
                plt.tight_layout()
                recon_spec_file = Path(output_dir) / f"{base_name}_reconstructed_spectrogram.png"
                plt.savefig(recon_spec_file, dpi=150, bbox_inches='tight')
                plt.close()
                viz_files['recon_spectrogram'] = str(recon_spec_file)
                del D_recon
                gc.collect()
                
                # Reconstructed mel spectrogram
                fig, ax = plt.subplots(1, 1, figsize=(8, 5))
                mel_recon = librosa.feature.melspectrogram(y=recon_audio, sr=self.sample_rate)
                mel_db_recon = librosa.power_to_db(mel_recon, ref=np.max)
                img4 = librosa.display.specshow(mel_db_recon, y_axis='mel', x_axis='time',
                                              sr=self.sample_rate, ax=ax)
                ax.set_title('Reconstructed Mel Spectrogram')
                plt.colorbar(img4, ax=ax, format='%+2.0f dB')
                plt.tight_layout()
                recon_mel_file = Path(output_dir) / f"{base_name}_reconstructed_mel_spectrogram.png"
                plt.savefig(recon_mel_file, dpi=150, bbox_inches='tight')
                plt.close()
                viz_files['recon_mel_spectrogram'] = str(recon_mel_file)
                del mel_recon, mel_db_recon, recon_audio
                gc.collect()
            
            # 3. Token distribution histograms
            print("  Generating token distributions...")
            semantic_codes = result.get('semantic_codes', [])
            acoustic_codes = result.get('acoustic_codes', [])
            
            if semantic_codes and acoustic_codes:
                # Semantic distributions
                for i, codes in enumerate(semantic_codes[:2]):
                    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
                    codes_flat = codes.flatten().cpu().numpy()
                    ax.hist(codes_flat, bins=50, alpha=0.7, edgecolor='black')
                    ax.set_title(f'Semantic Layer {i} Token Distribution')
                    ax.set_xlabel('Token ID')
                    ax.set_ylabel('Frequency')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    sem_dist_file = Path(output_dir) / f"{base_name}_semantic_layer_{i}_distribution.png"
                    plt.savefig(sem_dist_file, dpi=150, bbox_inches='tight')
                    plt.close()
                    viz_files[f'semantic_dist_{i}'] = str(sem_dist_file)
                    del codes_flat
                    gc.collect()
                
                # Acoustic distributions
                for i, codes in enumerate(acoustic_codes[:2]):
                    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
                    codes_flat = codes.flatten().cpu().numpy()
                    ax.hist(codes_flat, bins=50, alpha=0.7, edgecolor='black')
                    ax.set_title(f'Acoustic Layer {i} Token Distribution')
                    ax.set_xlabel('Token ID')
                    ax.set_ylabel('Frequency')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    ac_dist_file = Path(output_dir) / f"{base_name}_acoustic_layer_{i}_distribution.png"
                    plt.savefig(ac_dist_file, dpi=150, bbox_inches='tight')
                    plt.close()
                    viz_files[f'acoustic_dist_{i}'] = str(ac_dist_file)
                    del codes_flat
                    gc.collect()
            
            # 4. Feature representation heatmaps
            print("  Generating feature heatmaps...")
            semantic_features = result.get('semantic_features')
            acoustic_features = result.get('acoustic_features')
            
            if semantic_features is not None:
                fig, ax = plt.subplots(1, 1, figsize=(15, 5))
                sem_feat = semantic_features.squeeze().cpu().numpy()
                im1 = ax.imshow(sem_feat, aspect='auto', origin='lower', cmap='viridis')
                ax.set_title('Semantic Features Representation')
                ax.set_ylabel('Feature Dimension')
                ax.set_xlabel('Time Frame')
                plt.colorbar(im1, ax=ax)
                plt.tight_layout()
                sem_feat_file = Path(output_dir) / f"{base_name}_semantic_features.png"
                plt.savefig(sem_feat_file, dpi=150, bbox_inches='tight')
                plt.close()
                viz_files['semantic_features'] = str(sem_feat_file)
                del sem_feat
                gc.collect()
            
            if acoustic_features is not None:
                fig, ax = plt.subplots(1, 1, figsize=(15, 5))
                ac_feat = acoustic_features.squeeze().cpu().numpy()
                im2 = ax.imshow(ac_feat, aspect='auto', origin='lower', cmap='plasma')
                ax.set_title('Acoustic Features Representation')
                ax.set_xlabel('Time Frame')
                ax.set_ylabel('Feature Dimension')
                plt.colorbar(im2, ax=ax)
                plt.tight_layout()
                ac_feat_file = Path(output_dir) / f"{base_name}_acoustic_features.png"
                plt.savefig(ac_feat_file, dpi=150, bbox_inches='tight')
                plt.close()
                viz_files['acoustic_features'] = str(ac_feat_file)
                del ac_feat
                gc.collect()
        
        except Exception as e:
            print(f"Warning: Could not generate some visualizations: {e}")
        
        return viz_files
    
    def _generate_visualizations_parallel(self, original_audio: np.ndarray, result: Dict, output_dir: str, base_name: str):
        """Generate visualizations all at once (original method)."""
        viz_files = {}
        
        try:
            # Setup matplotlib for clean plots
            plt.style.use('default')
            
            # 1. Original vs Reconstructed Waveforms
            if result.get('reconstructed') is not None:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
                
                # Original waveform
                time_orig = np.linspace(0, len(original_audio) / self.sample_rate, len(original_audio))
                ax1.plot(time_orig, original_audio, 'b-', alpha=0.7, linewidth=0.5)
                ax1.set_title('Original Waveform')
                ax1.set_ylabel('Amplitude')
                ax1.grid(True, alpha=0.3)
                
                # Reconstructed waveform
                recon_audio = result['reconstructed'].squeeze().cpu().numpy()
                time_recon = np.linspace(0, len(recon_audio) / self.sample_rate, len(recon_audio))
                ax2.plot(time_recon, recon_audio, 'r-', alpha=0.7, linewidth=0.5)
                ax2.set_title('Reconstructed Waveform')
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Amplitude')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                waveform_file = Path(output_dir) / f"{base_name}_waveforms.png"
                plt.savefig(waveform_file, dpi=150, bbox_inches='tight')
                plt.close()
                viz_files['waveforms'] = str(waveform_file)
            
            # 2. Spectrograms comparison
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            
            # Original spectrogram
            D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(original_audio)), ref=np.max)
            img1 = librosa.display.specshow(D_orig, y_axis='hz', x_axis='time', 
                                          sr=self.sample_rate, ax=axes[0, 0])
            axes[0, 0].set_title('Original Spectrogram')
            plt.colorbar(img1, ax=axes[0, 0], format='%+2.0f dB')
            
            # Original mel spectrogram
            mel_orig = librosa.feature.melspectrogram(y=original_audio, sr=self.sample_rate)
            mel_db_orig = librosa.power_to_db(mel_orig, ref=np.max)
            img2 = librosa.display.specshow(mel_db_orig, y_axis='mel', x_axis='time',
                                          sr=self.sample_rate, ax=axes[0, 1])
            axes[0, 1].set_title('Original Mel Spectrogram')
            plt.colorbar(img2, ax=axes[0, 1], format='%+2.0f dB')
            
            if result.get('reconstructed') is not None:
                recon_audio = result['reconstructed'].squeeze().cpu().numpy()
                
                # Reconstructed spectrogram
                D_recon = librosa.amplitude_to_db(np.abs(librosa.stft(recon_audio)), ref=np.max)
                img3 = librosa.display.specshow(D_recon, y_axis='hz', x_axis='time',
                                              sr=self.sample_rate, ax=axes[1, 0])
                axes[1, 0].set_title('Reconstructed Spectrogram')
                plt.colorbar(img3, ax=axes[1, 0], format='%+2.0f dB')
                
                # Reconstructed mel spectrogram
                mel_recon = librosa.feature.melspectrogram(y=recon_audio, sr=self.sample_rate)
                mel_db_recon = librosa.power_to_db(mel_recon, ref=np.max)
                img4 = librosa.display.specshow(mel_db_recon, y_axis='mel', x_axis='time',
                                              sr=self.sample_rate, ax=axes[1, 1])
                axes[1, 1].set_title('Reconstructed Mel Spectrogram')
                plt.colorbar(img4, ax=axes[1, 1], format='%+2.0f dB')
            else:
                axes[1, 0].text(0.5, 0.5, 'No Reconstruction Available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 1].text(0.5, 0.5, 'No Reconstruction Available',
                               ha='center', va='center', transform=axes[1, 1].transAxes)
            
            plt.tight_layout()
            spectrogram_file = Path(output_dir) / f"{base_name}_spectrograms.png"
            plt.savefig(spectrogram_file, dpi=150, bbox_inches='tight')
            plt.close()
            viz_files['spectrograms'] = str(spectrogram_file)
            
            # 3. Token distribution histograms
            semantic_codes = result.get('semantic_codes', [])
            acoustic_codes = result.get('acoustic_codes', [])
            
            if semantic_codes and acoustic_codes:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                # Semantic token distributions
                for i, codes in enumerate(semantic_codes[:2]):  # Show first 2 layers
                    ax = axes[0, i] if i < 2 else None
                    if ax is not None:
                        codes_flat = codes.flatten().cpu().numpy()
                        ax.hist(codes_flat, bins=50, alpha=0.7, edgecolor='black')
                        ax.set_title(f'Semantic Layer {i} Token Distribution')
                        ax.set_xlabel('Token ID')
                        ax.set_ylabel('Frequency')
                        ax.grid(True, alpha=0.3)
                
                # Acoustic token distributions
                for i, codes in enumerate(acoustic_codes[:2]):  # Show first 2 layers
                    ax = axes[1, i] if i < 2 else None
                    if ax is not None:
                        codes_flat = codes.flatten().cpu().numpy()
                        ax.hist(codes_flat, bins=50, alpha=0.7, edgecolor='black')
                        ax.set_title(f'Acoustic Layer {i} Token Distribution')
                        ax.set_xlabel('Token ID')
                        ax.set_ylabel('Frequency')
                        ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                token_dist_file = Path(output_dir) / f"{base_name}_token_distributions.png"
                plt.savefig(token_dist_file, dpi=150, bbox_inches='tight')
                plt.close()
                viz_files['token_distributions'] = str(token_dist_file)
            
            # 4. Feature representation heatmaps
            semantic_features = result.get('semantic_features')
            acoustic_features = result.get('acoustic_features')
            
            if semantic_features is not None and acoustic_features is not None:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
                
                # Semantic features heatmap
                sem_feat = semantic_features.squeeze().cpu().numpy()
                im1 = ax1.imshow(sem_feat, aspect='auto', origin='lower', cmap='viridis')
                ax1.set_title('Semantic Features Representation')
                ax1.set_ylabel('Feature Dimension')
                plt.colorbar(im1, ax=ax1)
                
                # Acoustic features heatmap
                ac_feat = acoustic_features.squeeze().cpu().numpy()
                im2 = ax2.imshow(ac_feat, aspect='auto', origin='lower', cmap='plasma')
                ax2.set_title('Acoustic Features Representation')
                ax2.set_xlabel('Time Frame')
                ax2.set_ylabel('Feature Dimension')
                plt.colorbar(im2, ax=ax2)
                
                plt.tight_layout()
                features_file = Path(output_dir) / f"{base_name}_feature_maps.png"
                plt.savefig(features_file, dpi=150, bbox_inches='tight')
                plt.close()
                viz_files['feature_maps'] = str(features_file)
        
        except Exception as e:
            print(f"Warning: Could not generate some visualizations: {e}")
        
        return viz_files

    def save_detailed_analysis(self, original_audio: np.ndarray, result: Dict, 
                             metrics: TokenizationMetrics, output_dir: str, base_name: str):
        """Save detailed analysis data to various file formats."""
        analysis_files = {}
        
        try:
            # 1. Save raw feature arrays as .npy files
            if result.get('semantic_features') is not None:
                semantic_feat_file = Path(output_dir) / f"{base_name}_semantic_features.npy"
                np.save(semantic_feat_file, result['semantic_features'].cpu().numpy())
                analysis_files['semantic_features_npy'] = str(semantic_feat_file)
            
            if result.get('acoustic_features') is not None:
                acoustic_feat_file = Path(output_dir) / f"{base_name}_acoustic_features.npy"
                np.save(acoustic_feat_file, result['acoustic_features'].cpu().numpy())
                analysis_files['acoustic_features_npy'] = str(acoustic_feat_file)
            
            # 2. Save token sequences as .npy files
            if result.get('semantic_codes'):
                for i, codes in enumerate(result['semantic_codes']):
                    codes_file = Path(output_dir) / f"{base_name}_semantic_codes_layer_{i}.npy"
                    np.save(codes_file, codes.cpu().numpy())
                    analysis_files[f'semantic_codes_layer_{i}_npy'] = str(codes_file)
            
            if result.get('acoustic_codes'):
                for i, codes in enumerate(result['acoustic_codes']):
                    codes_file = Path(output_dir) / f"{base_name}_acoustic_codes_layer_{i}.npy"
                    np.save(codes_file, codes.cpu().numpy())
                    analysis_files[f'acoustic_codes_layer_{i}_npy'] = str(codes_file)
            
            # 3. Save audio analysis features
            # MFCC features
            mfcc = librosa.feature.mfcc(y=original_audio, sr=self.sample_rate, n_mfcc=13)
            mfcc_file = Path(output_dir) / f"{base_name}_mfcc.npy"
            np.save(mfcc_file, mfcc)
            analysis_files['mfcc_npy'] = str(mfcc_file)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=original_audio, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=original_audio, sr=self.sample_rate)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(original_audio)[0]
            
            spectral_features = {
                'spectral_centroids': spectral_centroids.tolist(),
                'spectral_rolloff': spectral_rolloff.tolist(),
                'zero_crossing_rate': zero_crossing_rate.tolist()
            }
            
            spectral_file = Path(output_dir) / f"{base_name}_spectral_features.json"
            with open(spectral_file, 'w') as f:
                json.dump(spectral_features, f, indent=2)
            analysis_files['spectral_features_json'] = str(spectral_file)
            
            # 4. Save comprehensive metrics as CSV
            import csv
            metrics_csv_file = Path(output_dir) / f"{base_name}_detailed_metrics.csv"
            
            with open(metrics_csv_file, 'w', newline='') as csvfile:
                fieldnames = ['metric', 'value', 'category']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                metrics_dict = asdict(metrics)
                for metric_name, value in metrics_dict.items():
                    category = 'unknown'
                    if 'token' in metric_name.lower():
                        category = 'tokenization'
                    elif 'loss' in metric_name.lower() or 'accuracy' in metric_name.lower():
                        category = 'reconstruction'
                    elif 'entropy' in metric_name.lower() or 'information' in metric_name.lower():
                        category = 'information_theory'
                    elif 'time' in metric_name.lower() or 'second' in metric_name.lower():
                        category = 'performance'
                    
                    writer.writerow({
                        'metric': metric_name,
                        'value': value,
                        'category': category
                    })
            
            analysis_files['detailed_metrics_csv'] = str(metrics_csv_file)
            
            # 5. Token sequence analysis
            if result.get('semantic_codes') and result.get('acoustic_codes'):
                # Token usage statistics
                token_stats = {}
                
                # Semantic token usage
                for i, codes in enumerate(result['semantic_codes']):
                    codes_flat = codes.flatten().cpu().numpy()
                    unique_tokens, counts = np.unique(codes_flat, return_counts=True)
                    token_stats[f'semantic_layer_{i}'] = {
                        'unique_tokens': len(unique_tokens),
                        'total_tokens': len(codes_flat),
                        'most_frequent_token': int(unique_tokens[np.argmax(counts)]),
                        'usage_entropy': float(entropy(counts / len(codes_flat)))
                    }
                
                # Acoustic token usage
                for i, codes in enumerate(result['acoustic_codes']):
                    codes_flat = codes.flatten().cpu().numpy()
                    unique_tokens, counts = np.unique(codes_flat, return_counts=True)
                    token_stats[f'acoustic_layer_{i}'] = {
                        'unique_tokens': len(unique_tokens),
                        'total_tokens': len(codes_flat),
                        'most_frequent_token': int(unique_tokens[np.argmax(counts)]),
                        'usage_entropy': float(entropy(counts / len(codes_flat)))
                    }
                
                token_stats_file = Path(output_dir) / f"{base_name}_token_statistics.json"
                with open(token_stats_file, 'w') as f:
                    json.dump(token_stats, f, indent=2)
                analysis_files['token_statistics_json'] = str(token_stats_file)
        
        except Exception as e:
            print(f"Warning: Could not save some analysis files: {e}")
        
        return analysis_files


# ============================================================================
# Token Format Converters & Streaming Protocols
# ============================================================================

class TokenFormatter:
    """Convert neural tokens to various LLM-friendly formats."""
    
    @staticmethod
    def to_text_sequence(semantic_codes: List[torch.Tensor], 
                        acoustic_codes: List[torch.Tensor],
                        format_type: str = "hierarchical") -> str:
        """Convert tokens to text sequence for LLM consumption."""
        
        if format_type == "hierarchical":
            return TokenFormatter._hierarchical_format(semantic_codes, acoustic_codes)
        elif format_type == "interleaved":
            return TokenFormatter._interleaved_format(semantic_codes, acoustic_codes)
        elif format_type == "structured":
            return TokenFormatter._structured_format(semantic_codes, acoustic_codes)
        else:
            raise ValueError(f"Unknown format type: {format_type}")
    
    @staticmethod
    def _hierarchical_format(semantic_codes, acoustic_codes) -> str:
        """Hierarchical format: semantic tokens first, then acoustic details."""
        text_tokens = []
        
        # Semantic layer
        text_tokens.append("[SEMANTIC]")
        for layer_idx, codes in enumerate(semantic_codes):
            layer_tokens = [f"S{layer_idx}_{int(code)}" for code in codes.flatten()]
            text_tokens.extend(layer_tokens)
        
        # Acoustic layer
        text_tokens.append("[ACOUSTIC]") 
        for layer_idx, codes in enumerate(acoustic_codes):
            layer_tokens = [f"A{layer_idx}_{int(code)}" for code in codes.flatten()]
            text_tokens.extend(layer_tokens)
        
        return " ".join(text_tokens)
    
    @staticmethod
    def _interleaved_format(semantic_codes, acoustic_codes) -> str:
        """Interleaved format: alternate semantic and acoustic tokens."""
        text_tokens = []
        
        # Flatten all codes
        all_semantic = torch.cat([codes.flatten() for codes in semantic_codes]) if semantic_codes else torch.tensor([])
        all_acoustic = torch.cat([codes.flatten() for codes in acoustic_codes]) if acoustic_codes else torch.tensor([])
        
        # Interleave tokens
        max_len = max(len(all_semantic), len(all_acoustic))
        for i in range(max_len):
            if i < len(all_semantic):
                text_tokens.append(f"S_{int(all_semantic[i])}")
            if i < len(all_acoustic):
                text_tokens.append(f"A_{int(all_acoustic[i])}")
        
        return " ".join(text_tokens)
    
    @staticmethod
    def _structured_format(semantic_codes, acoustic_codes) -> str:
        """Structured format with explicit timing and layering."""
        segments = []
        
        # Combine semantic and acoustic codes by time
        min_time_steps = min(
            min(codes.shape[-1] for codes in semantic_codes) if semantic_codes else 0,
            min(codes.shape[-1] for codes in acoustic_codes) if acoustic_codes else 0
        )
        
        for t in range(min_time_steps):
            segment = f"[T{t}]"
            
            # Semantic tokens for this timestep
            sem_tokens = []
            for layer_idx, codes in enumerate(semantic_codes):
                if t < codes.shape[-1]:
                    sem_tokens.append(f"S{layer_idx}:{int(codes[0, t])}")
            
            # Acoustic tokens for this timestep
            ac_tokens = []
            for layer_idx, codes in enumerate(acoustic_codes):
                if t < codes.shape[-1]:
                    ac_tokens.append(f"A{layer_idx}:{int(codes[0, t])}")
            
            segment += "[SEM:" + ",".join(sem_tokens) + "]"
            segment += "[AC:" + ",".join(ac_tokens) + "]"
            segments.append(segment)
        
        return " ".join(segments)
    
    @staticmethod
    def to_json(semantic_codes: List[torch.Tensor],
               acoustic_codes: List[torch.Tensor],
               metadata: Optional[Dict] = None) -> str:
        """Convert tokens to structured JSON format."""
        
        def tensor_to_list(tensor):
            return tensor.cpu().numpy().tolist()
        
        data = {
            "format_version": "1.0",
            "tokenization_type": "neural_hybrid",
            "semantic_tokens": {
                f"layer_{i}": tensor_to_list(codes)
                for i, codes in enumerate(semantic_codes)
            },
            "acoustic_tokens": {
                f"layer_{i}": tensor_to_list(codes) 
                for i, codes in enumerate(acoustic_codes)
            },
            "metadata": metadata or {}
        }
        
        return json.dumps(data, indent=2)


class StreamingProtocol:
    """Enhanced streaming protocol with optimized RLE, keyframes, and per-layer encoding."""
    
    def __init__(self, chunk_size: int = 8192, overlap: int = 1024, 
                 sample_rate: int = 22050, hop_length: int = 512,
                 rle_mode: bool = False, model_id: str = "tims-ears-v1.0",
                 codebook_size: int = 1024, num_semantic_layers: int = 4, 
                 num_acoustic_layers: int = 4, per_layer_encoding: Optional[Dict[str, str]] = None,
                 keyframe_interval_seconds: float = 5.0, audio_sha256: Optional[str] = None,
                 include_legend: bool = True):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.rle_mode = rle_mode
        self.keyframe_interval_seconds = keyframe_interval_seconds
        
        # Set up per-layer encoding with smart defaults
        if per_layer_encoding is None and rle_mode:
            # Default: RLE for semantic layers, dense for acoustic layers
            per_layer_encoding = {}
            for i in range(num_semantic_layers):
                per_layer_encoding[f"S{i}"] = "rle"
            for i in range(num_acoustic_layers):
                per_layer_encoding[f"A{i}"] = "dense"
        
        self.ndjson_streamer = NDJSONStreamer(
            sample_rate, hop_length, model_id, codebook_size,
            num_semantic_layers, num_acoustic_layers, rle_mode,
            per_layer_encoding, keyframe_interval_seconds, audio_sha256
        )
        
        # Track previous tokens for RLE change detection
        self.prev_semantic_tokens = None
        self.prev_acoustic_tokens = None
        self.last_keyframe_time = 0.0
        
    def create_stream_header(self, sample_rate: int, total_samples: int, metadata: Dict = None) -> str:
        """Create stream header with metadata."""
        header = {
            "stream_type": "neural_audio_tokens",
            "version": "1.0",
            "sample_rate": sample_rate,
            "total_samples": total_samples,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        
        return f"===STREAM_HEADER===\n{json.dumps(header)}\n===STREAM_START==="
    
    def create_chunk_marker(self, chunk_idx: int, chunk_samples: int, tokens: Dict) -> str:
        """Create chunk marker with token data."""
        chunk_data = {
            "chunk_index": chunk_idx,
            "chunk_samples": chunk_samples,
            "tokens": {
                "semantic": [[int(x) for x in codes.flatten()] for codes in tokens['semantic_codes']],
                "acoustic": [[int(x) for x in codes.flatten()] for codes in tokens['acoustic_codes']]
            }
        }
        
        return f"===CHUNK_{chunk_idx}===\n{json.dumps(chunk_data)}\n===CHUNK_END==="
    
    def create_stream_footer(self, processing_stats: Dict = None) -> str:
        """Create stream footer with processing statistics."""
        footer = {
            "processing_complete": True,
            "stats": processing_stats or {},
            "timestamp": time.time()
        }
        
        return f"===STREAM_FOOTER===\n{json.dumps(footer)}\n===STREAM_COMPLETE==="
    
    def _detect_changed_layers(self, semantic_tokens: List[int], acoustic_tokens: List[int]) -> List[str]:
        """Detect which layers changed from previous frame (for RLE mode)."""
        changed_layers = []
        
        # Compare semantic layers
        if self.prev_semantic_tokens is not None:
            for i, (curr, prev) in enumerate(zip(semantic_tokens, self.prev_semantic_tokens)):
                if curr != prev:
                    changed_layers.append(f"S{i}")
        else:
            # First frame - all layers are "changed"
            changed_layers.extend([f"S{i}" for i in range(len(semantic_tokens))])
            
        # Compare acoustic layers
        if self.prev_acoustic_tokens is not None:
            for i, (curr, prev) in enumerate(zip(acoustic_tokens, self.prev_acoustic_tokens)):
                if curr != prev:
                    changed_layers.append(f"A{i}")
        else:
            # First frame - all layers are "changed" 
            changed_layers.extend([f"A{i}" for i in range(len(acoustic_tokens))])
            
        # Update previous tokens
        self.prev_semantic_tokens = semantic_tokens[:]
        self.prev_acoustic_tokens = acoustic_tokens[:]
        
        return changed_layers
    
    def _should_emit_keyframe(self, time_seconds: float) -> bool:
        """Determine if we should emit a keyframe at this time."""
        if not self.rle_mode:
            return False
        
        if time_seconds - self.last_keyframe_time >= self.keyframe_interval_seconds:
            self.last_keyframe_time = time_seconds
            return True
        return False
    
    def create_ndjson_stream(self, tokens: Dict, metadata: Dict = None, 
                           processing_stats: Dict = None, duration_seconds: float = None,
                           include_legend: bool = True) -> str:
        """Create optimized NDJSON stream with RLE aggregation, keyframes, and per-layer encoding."""
        lines = []
        
        # Enhanced header with duration
        lines.append(self.ndjson_streamer.create_header(duration_seconds, metadata, include_legend))
        
        # Frame data with accurate timebase and optimizations
        semantic_codes = tokens['semantic_codes']
        acoustic_codes = tokens['acoustic_codes']
        
        if semantic_codes and acoustic_codes:
            # Determine frame count from actual token arrays
            num_frames = min(
                min(codes.shape[-1] for codes in semantic_codes),
                min(codes.shape[-1] for codes in acoustic_codes)
            )
            
            # Use accurate frame timing
            frames_per_second = self.ndjson_streamer.frames_per_second
            
            # Reset state for new stream
            self.prev_semantic_tokens = None
            self.prev_acoustic_tokens = None
            self.last_keyframe_time = 0.0
            self.ndjson_streamer.buffered_event = None
            self.ndjson_streamer.last_frame_index = -1
            
            for frame_idx in range(num_frames):
                # Extract tokens for this frame
                sem_tokens = [int(codes[0, frame_idx]) for codes in semantic_codes 
                            if frame_idx < codes.shape[-1]]
                acc_tokens = [int(codes[0, frame_idx]) for codes in acoustic_codes 
                            if frame_idx < codes.shape[-1]]
                
                time_ms = frame_idx * self.ndjson_streamer.frame_duration_ms
                time_seconds = time_ms / 1000.0
                
                # Check for keyframe
                is_keyframe = self._should_emit_keyframe(time_seconds)
                
                if self.rle_mode and not is_keyframe:
                    # RLE mode: detect changes and use aggregation
                    changed_layers = self._detect_changed_layers(sem_tokens, acc_tokens)
                    
                    frame_output = self.ndjson_streamer.create_frame(
                        frame_idx, time_ms, sem_tokens, acc_tokens, 
                        changed_layers=changed_layers, is_keyframe=False
                    )
                else:
                    # Dense mode or keyframe: emit full frame
                    frame_output = self.ndjson_streamer.create_frame(
                        frame_idx, time_ms, sem_tokens, acc_tokens,
                        is_keyframe=is_keyframe
                    )
                    # Update change tracking for dense frames too
                    if self.rle_mode:
                        self._detect_changed_layers(sem_tokens, acc_tokens)
                
                if frame_output:
                    lines.append(frame_output)
        
        # End marker with final flush and stats
        end_output = self.ndjson_streamer.create_end_marker(processing_stats)
        lines.append(end_output)
        
        return "\n".join(lines)


# ============================================================================
# Main Audio Processing Pipeline
# ============================================================================

class AudioTokenizationPipeline:
    """Main pipeline for audio tokenization with scientific evaluation."""
    
    def __init__(self,
                 sample_rate: int = 22050,
                 model_config: Optional[Dict] = None,
                 device: str = "auto",
                 enable_compat_fallback: bool = True,
                 resample_rate: Optional[int] = None,
                 rle_mode: bool = False,
                 model_id: str = "tims-ears-v1.0",
                 per_layer_encoding: Optional[Dict[str, str]] = None,
                 keyframe_interval_seconds: float = 5.0,
                 include_legend: bool = True):
        self.original_sample_rate = sample_rate  # Keep track of what was requested
        self.resample_rate = resample_rate  # None means no resampling
        self.sample_rate = resample_rate if resample_rate is not None else sample_rate
        self.model_config = model_config or {}
        self.enable_compat_fallback = enable_compat_fallback
        self.rle_mode = rle_mode
        self.model_id = model_id
        self.per_layer_encoding = per_layer_encoding
        self.keyframe_interval_seconds = keyframe_interval_seconds
        self.include_legend = include_legend
        
        # Device selection with improved fallback handling
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Check dependencies and set compat mode if needed
        self.compat_mode = not self._check_dependencies()
        
        if self.compat_mode and enable_compat_fallback:
            print("Warning: Falling back to compatibility mode (some features may be limited)")
            # Initialize with basic fallbacks
            self.tokenizer = None  # Will use compat tokenizer
        else:
            # Initialize full neural tokenizer
            hop_length = self.model_config.get('hop_length', 512)
            # Remove hop_length from model_config to avoid duplicate argument
            tokenizer_config = {k: v for k, v in self.model_config.items() if k != 'hop_length'}
            self.tokenizer = NeuralAudioTokenizer(
                sample_rate=sample_rate,
                hop_length=hop_length,
                **tokenizer_config
            ).to(self.device)
        
        # Initialize evaluator
        self.evaluator = TokenizationEvaluator(sample_rate)
        
        # Token formatter
        self.formatter = TokenFormatter()
        
        # Enhanced streaming protocol with hop length and model info
        hop_length = self.model_config.get('hop_length', 512)
        codebook_size = self.model_config.get('codebook_size', 1024)
        num_quantizers = self.model_config.get('num_quantizers', 8)
        
        # Generate audio SHA256 if available
        self.audio_sha256 = None  # Will be set per file
        
        self.streaming = StreamingProtocol(
            sample_rate=sample_rate, 
            hop_length=hop_length,
            rle_mode=rle_mode,
            model_id=model_id,
            codebook_size=codebook_size,
            num_semantic_layers=num_quantizers//2,
            num_acoustic_layers=num_quantizers//2,
            per_layer_encoding=per_layer_encoding,
            keyframe_interval_seconds=keyframe_interval_seconds,
            audio_sha256=self.audio_sha256,
            include_legend=include_legend
        )
        
        # Token budget meter with accurate timing
        self.budget_meter = TokenBudgetMeter(sample_rate, hop_length)
        
        print(f"Initialized Neural Audio Tokenizer on {self.device}")
        print(f"Model ID: {model_id}, RLE Mode: {rle_mode}")
        if per_layer_encoding:
            print(f"Per-layer encoding: {per_layer_encoding}")
        if self.compat_mode:
            print("Running in compatibility mode")
    
    def _generate_audio_sha256(self, audio: np.ndarray) -> str:
        """Generate SHA256 hash of audio data for integrity checking."""
        import hashlib
        audio_bytes = audio.astype(np.float32).tobytes()
        return hashlib.sha256(audio_bytes).hexdigest()
        
    def _check_dependencies(self) -> bool:
        """Check if all required dependencies are available."""
        try:
            import torch
            import torchaudio
            import librosa 
            import soundfile
            return True
        except ImportError:
            return False
    
    def load_audio(self, file_path: str, target_length: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Load and preprocess audio file with improved fallback handling and optional resampling."""
        audio = None
        original_sr = None
        
        # Try multiple audio loading backends
        loaders = []
        
        # Try librosa first (most reliable)
        if 'librosa' in sys.modules or self._try_import('librosa'):
            loaders.append(('librosa', self._load_with_librosa))
        
        # Try torchaudio
        if 'torchaudio' in sys.modules or self._try_import('torchaudio'):
            loaders.append(('torchaudio', self._load_with_torchaudio))
        
        # Try soundfile
        if 'soundfile' in sys.modules or self._try_import('soundfile'):
            loaders.append(('soundfile', self._load_with_soundfile))
        
        # Try each loader until one succeeds
        last_error = None
        for loader_name, loader_func in loaders:
            try:
                audio, original_sr = loader_func(file_path)
                break
            except Exception as e:
                last_error = e
                continue
        
        if audio is None:
            # Last resort: try to read as raw audio bytes
            try:
                audio = self._load_as_raw_bytes(file_path)
                original_sr = self.original_sample_rate  # Fallback assumption
            except Exception as e:
                raise RuntimeError(f"Could not load audio file {file_path}. Tried all available backends. Last error: {last_error}")
        
        # Resample only if --resample flag was used
        final_sr = original_sr
        if self.resample_rate is not None:
            if self.resample_rate <= 0:
                # Use default 22050 for 0 or negative values
                target_sr = 22050
            else:
                target_sr = self.resample_rate
            
            if original_sr != target_sr:
                import librosa
                audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
                final_sr = target_sr
                print(f"Resampled from {original_sr} Hz to {target_sr} Hz")
        
        # Normalize
        audio = audio / (np.abs(audio).max() + 1e-8)
        
        # Truncate or pad to target length
        if target_length is not None:
            if len(audio) > target_length:
                audio = audio[:target_length]
            elif len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        
        return audio, final_sr
    
    def _try_import(self, module_name: str) -> bool:
        """Try to import a module and return success."""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
    
    def _load_with_librosa(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio using librosa at native sample rate."""
        import librosa
        audio, sr = librosa.load(file_path, sr=None, mono=True)  # sr=None preserves original
        return audio, sr
    
    def _load_with_torchaudio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio using torchaudio at native sample rate."""
        import torchaudio
        
        audio_tensor, sr = torchaudio.load(file_path)
        audio = audio_tensor.mean(dim=0).numpy()  # Convert to mono
        return audio, sr
    
    def _load_with_soundfile(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio using soundfile at native sample rate."""
        import soundfile as sf
        
        audio, sr = sf.read(file_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Convert to mono
        return audio, sr
    
    def _load_as_raw_bytes(self, file_path: str) -> np.ndarray:
        """Load file as raw bytes and interpret as 16-bit PCM audio."""
        with open(file_path, 'rb') as f:
            raw_bytes = f.read()
        
        # Interpret as 16-bit little-endian PCM
        audio_int16 = np.frombuffer(raw_bytes, dtype='<i2')
        audio_float = audio_int16.astype(np.float32) / 32768.0
        
        return audio_float
    
    def process_audio(self,
                     file_path: str,
                     output_format: str = "hierarchical",
                     enable_reconstruction: bool = True,
                     streaming_mode: bool = False,
                     ndjson_streaming: bool = False) -> Dict[str, Any]:
        """Process audio file through complete tokenization pipeline."""
        
        print(f"Processing: {file_path}")
        start_time = time.time()
        
        # Reset budget meter
        self.budget_meter.reset()
        
        # Load audio
        audio, sr = self.load_audio(file_path)
        print(f"Loaded audio: {len(audio)} samples, {sr} Hz, {len(audio)/sr:.2f}s")
        
        # Generate audio integrity hash
        audio_hash = self._generate_audio_sha256(audio)
        
        # Update streaming protocol with audio hash
        self.streaming.ndjson_streamer.audio_sha256 = audio_hash
        
        # Convert to tensor and ensure on correct device
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        
        # Process through tokenizer
        print("Tokenizing...")
        with torch.no_grad():
            result = self.tokenizer(audio_tensor)
        
        semantic_codes = result['semantic_codes']
        acoustic_codes = result['acoustic_codes']
        reconstructed = result['reconstructed']
        num_frames = result.get('num_frames', 0)
        
        # Update budget meter
        num_sem_tokens = sum(codes.numel() for codes in semantic_codes)
        num_acc_tokens = sum(codes.numel() for codes in acoustic_codes)
        self.budget_meter.update(len(audio), num_frames, num_sem_tokens, num_acc_tokens)
        
        print(f"Generated {len(semantic_codes)} semantic layers, {len(acoustic_codes)} acoustic layers")
        print(f"Total tokens: {num_sem_tokens + num_acc_tokens}")
        
        # Evaluation
        print("Evaluating tokenization quality...")
        metrics = self.evaluator.evaluate_tokenization(
            audio, self.tokenizer, reconstructed
        )
        
        # Format tokens
        print("Formatting tokens...")
        text_tokens = self.formatter.to_text_sequence(
            semantic_codes, acoustic_codes, output_format
        )
        
        # Get budget metrics
        budget_metrics = self.budget_meter.get_metrics()
        
        json_tokens = self.formatter.to_json(
            semantic_codes, acoustic_codes,
            metadata={
                "file_path": file_path,
                "sample_rate": sr,
                "duration": len(audio) / sr,
                "processing_time": time.time() - start_time,
                "budget_metrics": asdict(budget_metrics),
                "audio_sha256": audio_hash,
                "model_id": self.model_id
            }
        )
        
        # Streaming formats
        streaming_output = None
        ndjson_output = None
        
        if streaming_mode:
            header = self.streaming.create_stream_header(sr, len(audio))
            chunk = self.streaming.create_chunk_marker(0, len(audio), result)
            footer = self.streaming.create_stream_footer({
                **asdict(metrics),
                **asdict(budget_metrics)
            })
            streaming_output = f"{header}\n{chunk}\n{footer}"
        
        if ndjson_streaming:
            ndjson_output = self.streaming.create_ndjson_stream(
                result,
                metadata={
                    "file_path": file_path,
                    "sample_rate": sr,
                    "duration": len(audio) / sr,
                    "audio_sha256": audio_hash,
                    "model_id": self.model_id
                },
                processing_stats={
                    **asdict(metrics),
                    **asdict(budget_metrics)
                },
                duration_seconds=len(audio) / sr,
                include_legend=self.include_legend
            )
        
        total_time = time.time() - start_time
        print(f"Processing complete in {total_time:.2f}s")
        print(f"Throughput: {budget_metrics.tokens_per_second:.1f} tokens/sec, {budget_metrics.frames_per_second:.1f} frames/sec")
        
        return {
            "semantic_codes": semantic_codes,
            "acoustic_codes": acoustic_codes,
            "text_tokens": text_tokens,
            "json_tokens": json_tokens,
            "streaming_output": streaming_output,
            "ndjson_output": ndjson_output,
            "reconstructed_audio": reconstructed.cpu().numpy() if reconstructed is not None else None,
            "metrics": metrics,
            "budget_metrics": budget_metrics,
            "processing_time": total_time,
            "original_audio": audio,  # Include original for analysis
            "tokenizer_result": result,  # Include full result for detailed analysis
            "metadata": {
                "file_path": file_path,
                "sample_rate": sr,
                "duration": len(audio) / sr,
                "device": str(self.device),
                "compat_mode": self.compat_mode,
                "audio_sha256": audio_hash,
                "model_id": self.model_id
            }
        }
    
    def batch_process(self, 
                     input_paths: List[str],
                     output_dir: str,
                     output_format: str = "hierarchical",
                     sequential_vis: bool = False) -> List[Dict]:
        """Batch process multiple audio files."""
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        for i, file_path in enumerate(input_paths):
            print(f"\nProcessing {i+1}/{len(input_paths)}: {file_path}")
            
            try:
                result = self.process_audio(file_path, output_format)
                
                # Save outputs
                base_name = Path(file_path).stem
                
                # Text tokens
                with open(Path(output_dir) / f"{base_name}_tokens.txt", 'w') as f:
                    f.write(result['text_tokens'])
                
                # JSON tokens
                with open(Path(output_dir) / f"{base_name}_tokens.json", 'w') as f:
                    f.write(result['json_tokens'])
                
                # Streaming format
                if result['streaming_output']:
                    with open(Path(output_dir) / f"{base_name}_stream.txt", 'w') as f:
                        f.write(result['streaming_output'])
                
                # Enhanced NDJSON format (always generate for batch processing)
                ndjson_output = self.streaming.create_ndjson_stream(
                    result['tokenizer_result'],
                    metadata={
                        "file_path": file_path,
                        "sample_rate": result['metadata']['sample_rate'],
                        "duration": result['metadata']['duration'],
                        "audio_sha256": result['metadata'].get('audio_sha256'),
                        "model_id": result['metadata'].get('model_id', self.model_id)
                    },
                    processing_stats={
                        **asdict(result['metrics']),
                        **asdict(result['budget_metrics'])
                    },
                    duration_seconds=result['metadata']['duration'],
                    include_legend=self.include_legend
                )
                
                with open(Path(output_dir) / f"{base_name}_tokens.ndjson", 'w') as f:
                    f.write(ndjson_output)
                
                # Reconstructed audio
                if result['reconstructed_audio'] is not None:
                    try:
                        import soundfile as sf
                        sf.write(
                            Path(output_dir) / f"{base_name}_reconstructed.wav",
                            result['reconstructed_audio'].squeeze(),
                            self.sample_rate
                        )
                    except:
                        print(f"Warning: Could not save reconstructed audio for {base_name}")
                
                # Metrics
                with open(Path(output_dir) / f"{base_name}_metrics.json", 'w') as f:
                    json.dump({
                        **asdict(result['metrics']),
                        **asdict(result['budget_metrics'])
                    }, f, indent=2)
                
                # Generate and save visualizations
                print(f"Generating visualizations for {base_name}...")
                viz_files = self.evaluator.generate_visualizations(
                    result['original_audio'], result['tokenizer_result'], output_dir, base_name, sequential=sequential_vis
                )
                
                # Save detailed analysis files
                print(f"Saving detailed analysis for {base_name}...")
                analysis_files = self.evaluator.save_detailed_analysis(
                    result['original_audio'], result['tokenizer_result'], 
                    result['metrics'], output_dir, base_name
                )
                
                # Add file listings to result
                result['generated_files'] = {
                    'visualizations': viz_files,
                    'analysis_files': analysis_files
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                results.append({"error": str(e), "file_path": file_path})
        
        return results


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Neural Audio-to-LLM Tokenizer - Research-grade music tokenization for language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s song.wav --output tokens.txt
  %(prog)s song.mp3 --format structured --streaming --all-outputs output_dir/
  %(prog)s --stdin --format interleaved > tokens.txt
  echo "song.wav" | %(prog)s --stdin --batch
  %(prog)s *.wav --batch --output-dir results/ --format hierarchical
  %(prog)s song.wav --evaluate --reconstruction --metrics metrics.json
  %(prog)s song.flac --streaming --chunk-size 16384 > stream.txt
  %(prog)s song.wav --ndjson-streaming > tokens.ndjson
  %(prog)s --resample 48000 song.wav  # Resample to 48kHz
  %(prog)s --resample song.wav        # Resample to default 22050Hz
        """
    )
    
    # Input/Output
    parser.add_argument('input_files', nargs='*', help='Input audio files')
    parser.add_argument('--stdin', action='store_true', help='Read file paths from stdin')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    parser.add_argument('--output-dir', help='Output directory for all outputs mode')
    parser.add_argument('--all-outputs', action='store_true', help='Generate all output formats')
    
    # Tokenization format
    parser.add_argument('--format', choices=['hierarchical', 'interleaved', 'structured'], 
                       default='hierarchical', help='Token format (default: hierarchical)')
    parser.add_argument('--streaming', action='store_true', help='Use streaming protocol output')
    parser.add_argument('--ndjson-streaming', action='store_true', help='Use NDJSON streaming (LAM v0.1)')
    parser.add_argument('--rle', action='store_true', help='Use RLE (run-length encoding) mode for more efficient NDJSON streaming')
    parser.add_argument('--chunk-size', type=int, default=8192, help='Streaming chunk size')
    parser.add_argument('--model-id', default='tims-ears-v1.0', help='Model identifier for token semantics stability (default: tims-ears-v1.0)')
    
    # Advanced RLE and encoding options
    parser.add_argument('--keyframe-interval', type=float, default=5.0, help='Keyframe interval in seconds for RLE mode (default: 5.0)')
    parser.add_argument('--encoding', help='Per-layer encoding specification, e.g., "S0=rle,S1=rle,A0=dense,A1=dense" or "S=rle,A=dense"')
    parser.add_argument('--rle-semantic', action='store_true', help='Force RLE encoding for all semantic layers')
    parser.add_argument('--dense-acoustic', action='store_true', help='Force dense encoding for all acoustic layers (default in RLE mode)')
    parser.add_argument('--no-legend', action='store_true', help='Omit legend from NDJSON header to save tokens')
    
    # Backward compatibility  
    parser.add_argument('--ndjson-compat', action='store_true', help='Use legacy NDJSON format for backward compatibility')
    
    # Model configuration - FIXED: Added proper --resample argument
    parser.add_argument('--sample-rate', type=int, default=22050, help='Target sample rate (deprecated, use --resample)')
    parser.add_argument('--resample', type=int, nargs='?', const=22050, default=None, help='Resample audio to specified Hz (default: no resampling, --resample alone uses 22050Hz)')
    parser.add_argument('--hop-length', type=int, default=512, help='STFT hop length')
    parser.add_argument('--semantic-dim', type=int, default=512, help='Semantic feature dimension')
    parser.add_argument('--acoustic-dim', type=int, default=512, help='Acoustic feature dimension') 
    parser.add_argument('--codebook-size', type=int, default=1024, help='Quantizer codebook size')
    parser.add_argument('--num-quantizers', type=int, default=8, help='Number of quantizer layers')
    parser.add_argument('--n-mels', type=int, default=128, help='Number of mel bands')
    
    # Evaluation
    parser.add_argument('--evaluate', action='store_true', help='Run comprehensive evaluation')
    parser.add_argument('--reconstruction', action='store_true', help='Enable audio reconstruction')
    parser.add_argument('--metrics', help='Output metrics to JSON file')
    parser.add_argument('--budget-report', action='store_true', help='Show detailed token budget report')
    parser.add_argument('--seq-vis', action='store_true', help='Use sequential visualization generation (slower but lower memory)')
    
    # Processing options
    parser.add_argument('--batch', action='store_true', help='Batch processing mode')
    parser.add_argument('--device', default='auto', help='Processing device (auto/cpu/cuda)')
    parser.add_argument('--max-length', type=int, help='Maximum audio length in samples')
    parser.add_argument('--compat-fallback', action='store_true', help='Enable compatibility fallback mode')
    
    # Advanced options
    parser.add_argument('--model-path', help='Path to pre-trained model')
    parser.add_argument('--config', help='Model configuration JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup
    if args.verbose:
        print("Enhanced Neural Audio-to-LLM Tokenizer v1.0")
        print(f"PyTorch: {torch.__version__}")
        print(f"Device: {args.device}")
    
    # Load model configuration
    model_config = {}
    if args.config:
        with open(args.config) as f:
            model_config = json.load(f)
    
    # Override with command line arguments
    model_config.update({
        'semantic_dim': args.semantic_dim,
        'acoustic_dim': args.acoustic_dim,
        'codebook_size': args.codebook_size,
        'num_quantizers': args.num_quantizers,
        'n_mels': args.n_mels,
        'hop_length': args.hop_length
    })
    
    # Determine resampling behavior - FIXED: Handle args.resample properly
    resample_rate = None
    if args.resample is not None:
        if args.resample <= 0:
            resample_rate = 22050  # Default for 0 or negative
        else:
            resample_rate = args.resample
    
    # Parse per-layer encoding specification
    per_layer_encoding = None
    if args.encoding:
        per_layer_encoding = {}
        
        # Handle shorthand notation like "S=rle,A=dense"
        if args.encoding in ["S=rle,A=dense", "S=dense,A=rle"]:
            for i in range(model_config.get('num_quantizers', 8) // 2):
                if "S=rle" in args.encoding:
                    per_layer_encoding[f"S{i}"] = "rle"
                if "A=dense" in args.encoding:
                    per_layer_encoding[f"A{i}"] = "dense"
                if "S=dense" in args.encoding:
                    per_layer_encoding[f"S{i}"] = "dense"
                if "A=rle" in args.encoding:
                    per_layer_encoding[f"A{i}"] = "rle"
        else:
            # Handle explicit layer specification like "S0=rle,S1=rle,A0=dense,A1=dense"
            for spec in args.encoding.split(','):
                if '=' in spec:
                    layer_name, encoding_type = spec.split('=')
                    if encoding_type in ['rle', 'dense']:
                        per_layer_encoding[layer_name.strip()] = encoding_type.strip()
    
    # Apply CLI flag overrides
    if args.rle_semantic or args.dense_acoustic:
        if per_layer_encoding is None:
            per_layer_encoding = {}
        
        num_quantizers = model_config.get('num_quantizers', 8)
        if args.rle_semantic:
            for i in range(num_quantizers // 2):
                per_layer_encoding[f"S{i}"] = "rle"
        if args.dense_acoustic:
            for i in range(num_quantizers // 2):
                per_layer_encoding[f"A{i}"] = "dense"
    
    # Initialize pipeline
    pipeline = AudioTokenizationPipeline(
        sample_rate=args.sample_rate,  # Keep for compatibility
        model_config=model_config,
        device=args.device,
        enable_compat_fallback=args.compat_fallback,
        resample_rate=resample_rate,
        rle_mode=args.rle,
        model_id=args.model_id,
        per_layer_encoding=per_layer_encoding,
        keyframe_interval_seconds=args.keyframe_interval,
        include_legend=not args.no_legend
    )
    
    # Get input files
    input_files = []
    if args.stdin:
        input_files = [line.strip() for line in sys.stdin if line.strip()]
    elif args.input_files:
        input_files = args.input_files
    else:
        parser.error("No input files provided. Use positional arguments or --stdin")
    
    # Validate input files
    for file_path in input_files:
        if not os.path.exists(file_path):
            parser.error(f"Input file not found: {file_path}")
    
    # Processing
    if args.batch or len(input_files) > 1:
        # Batch processing
        if not args.output_dir:
            parser.error("--output-dir required for batch processing")
        
        results = pipeline.batch_process(
            input_files, 
            args.output_dir, 
            args.format,
            sequential_vis=args.seq_vis
        )
        
        # Summary statistics
        successful = [r for r in results if 'error' not in r]
        failed = [r for r in results if 'error' in r]
        
        print(f"\nBatch processing complete:")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        
        # Report generated files
        if successful:
            total_viz_files = sum(len(r.get('generated_files', {}).get('visualizations', {})) for r in successful)
            total_analysis_files = sum(len(r.get('generated_files', {}).get('analysis_files', {})) for r in successful)
            print(f"  Generated {total_viz_files} visualization files")
            print(f"  Generated {total_analysis_files} analysis files")
        
        # Aggregate metrics
        if args.metrics and successful:
            avg_metrics = {}
            metric_keys = list(asdict(successful[0]['metrics']).keys())
            budget_keys = list(asdict(successful[0]['budget_metrics']).keys())
            
            for key in metric_keys + budget_keys:
                if key in metric_keys:
                    values = [asdict(r['metrics'])[key] for r in successful 
                             if isinstance(asdict(r['metrics'])[key], (int, float))]
                else:
                    values = [asdict(r['budget_metrics'])[key] for r in successful 
                             if isinstance(asdict(r['budget_metrics'])[key], (int, float))]
                
                if values:
                    avg_metrics[f"avg_{key}"] = sum(values) / len(values)
                    avg_metrics[f"min_{key}"] = min(values)
                    avg_metrics[f"max_{key}"] = max(values)
            
            with open(args.metrics, 'w') as f:
                json.dump(avg_metrics, f, indent=2)
        
    else:
        # Single file processing
        result = pipeline.process_audio(
            input_files[0],
            output_format=args.format,
            enable_reconstruction=args.reconstruction,
            streaming_mode=args.streaming,
            ndjson_streaming=args.ndjson_streaming
        )
        
        # Output results
        if args.all_outputs and args.output_dir:
            # Save all outputs
            os.makedirs(args.output_dir, exist_ok=True)
            base_name = Path(input_files[0]).stem
            
            # Text tokens
            with open(Path(args.output_dir) / f"{base_name}_tokens.txt", 'w') as f:
                f.write(result['text_tokens'])
            
            # JSON tokens  
            with open(Path(args.output_dir) / f"{base_name}_tokens.json", 'w') as f:
                f.write(result['json_tokens'])
            
            # Streaming format
            if result['streaming_output']:
                with open(Path(args.output_dir) / f"{base_name}_stream.txt", 'w') as f:
                    f.write(result['streaming_output'])
            
            # NDJSON format
            if result['ndjson_output']:
                with open(Path(args.output_dir) / f"{base_name}_ndjson.txt", 'w') as f:
                    f.write(result['ndjson_output'])
            
            # Reconstructed audio
            if result['reconstructed_audio'] is not None:
                try:
                    import soundfile as sf
                    sf.write(
                        Path(args.output_dir) / f"{base_name}_reconstructed.wav",
                        result['reconstructed_audio'].squeeze(),
                        pipeline.sample_rate
                    )
                except:
                    print(f"Warning: Could not save reconstructed audio")
            
            # Generate and save visualizations
            print(f"Generating visualizations...")
            viz_files = pipeline.evaluator.generate_visualizations(
                result['original_audio'], result['tokenizer_result'], args.output_dir, base_name, sequential=args.seq_vis
            )
            
            # Save detailed analysis files
            print(f"Saving detailed analysis...")
            analysis_files = pipeline.evaluator.save_detailed_analysis(
                result['original_audio'], result['tokenizer_result'], 
                result['metrics'], args.output_dir, base_name
            )
            
            # Report what was generated
            print(f"All outputs saved to: {args.output_dir}")
            if viz_files:
                print(f"Generated {len(viz_files)} visualization files")
            if analysis_files:
                print(f"Generated {len(analysis_files)} analysis files")
            
            output_text = None  # Don't print to stdout when using --all-outputs
            
        elif args.ndjson_streaming and result['ndjson_output']:
            output_text = result['ndjson_output']
        elif args.streaming and result['streaming_output']:
            output_text = result['streaming_output']
        else:
            # Regular token output
            output_text = result['text_tokens']
        
        # Write output
        if args.output:
            if output_text is not None:
                with open(args.output, 'w') as f:
                    f.write(output_text)
        else:
            if output_text is not None:
                print(output_text)
        
        # Metrics output
        if args.metrics:
            with open(args.metrics, 'w') as f:
                json.dump({
                    **asdict(result['metrics']),
                    **asdict(result['budget_metrics'])
                }, f, indent=2)
        
        # Budget report
        if args.budget_report:
            budget = result['budget_metrics']
            print(f"\nToken Budget Report:")
            print(f"  Total Tokens: {budget.total_tokens}")
            print(f"  Semantic Tokens: {budget.semantic_tokens}")
            print(f"  Acoustic Tokens: {budget.acoustic_tokens}")
            print(f"  Tokens/Second: {budget.tokens_per_second:.1f}")
            print(f"  Frames/Second: {budget.frames_per_second:.1f}")
            print(f"  Compression Ratio: {budget.compression_ratio:.1f}x")
        
        # Evaluation summary
        if args.evaluate:
            metrics = result['metrics']
            print(f"\nEvaluation Results:")
            print(f"  Compression Ratio: {metrics.compression_ratio:.2f}x")
            print(f"  Token Diversity: {metrics.token_diversity:.3f}")
            print(f"  Semantic Entropy: {metrics.semantic_entropy:.3f}")
            print(f"  Acoustic Entropy: {metrics.acoustic_entropy:.3f}")
            
            if args.reconstruction:
                print(f"  MSE Loss: {metrics.mse_loss:.6f}")
                print(f"  Spectral Loss: {metrics.spectral_loss:.6f}")
                print(f"  Pitch Accuracy: {metrics.pitch_accuracy:.3f}")
                print(f"  Rhythm Accuracy: {metrics.rhythm_accuracy:.3f}")
                print(f"  Timbral Similarity: {metrics.timbral_similarity:.3f}")


if __name__ == "__main__":
    main()
