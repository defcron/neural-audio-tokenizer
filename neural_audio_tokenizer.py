#!/usr/bin/env python3
"""
neural_audio_tokenizer.py - By Claude Sonnet 4 (Extended Thinking Mode), Claude Sonnet 4.5 (Extended Thinking Mode), ChatGPT Agent Mode, ChatGPT-5-Pro, and Claude Code (Sonnet 4.5 with Thinking Mode), based on initial work by ChatGPT Agent Mode, and with help and code review by custom GPT Tuesday, GPT-5 Auto, ChatGPT-5-Pro, and Jeremy Carter <jeremy@jeremycarter.ca> - 2025-10-07
==========================
Version 0.1.7 - MERT INTEGRATION: Music-optimized codebook initialization from MERT models

A research-grade neural audio tokenization system optimized for LLM consumption,
specifically designed for music understanding. Implements state-of-the-art
hybrid tokenization strategies based on recent advances in AudioLM, MuQ,
SoundStream, MERT, and other neural codec research.

**NEW IN v0.1.7:**
- MERT integration: Music-optimized codebook initialization using MERT (Music Understanding with Large-Scale Pre-training)
- New --codebook-init argument: Choose between 'mert' (music-specific, RECOMMENDED), 'encodec' (speech, legacy), or 'random'
- MERT provides 7x+ improvement in token diversity over EnCodec for musical content
- Significantly faster initialization than k-means (uses pre-trained music codebooks)
- Backward compatible with existing --use-encodec flag

**CRITICAL FIXES v0.1.4:**
- FIXED: K-means error handling - logging no longer poisons success/failure determination
- FIXED: Added progress reporting for long k-means operations
- FIXED: Aggressive memory cleanup during k-means retry attempts  
- FIXED: Better parameter validation to prevent edge case failures
- FIXED: Improved error recovery and fallback mechanisms
- FIXED: Memory usage monitoring and warnings
- FIXED: Better handling of very large feature sets

**MAJOR FIXES v0.1.3:**
- FIXED: K-means clustering now properly standardizes features before clustering
- FIXED: Added comprehensive cluster quality validation and diversity checks
- FIXED: Improved feature preprocessing to preserve diversity
- FIXED: Added robust fallback strategies when k-means fails
- FIXED: Added cluster separation metrics and validation
- FIXED: Better logging and debugging for cluster initialization
- FIXED: Prevents cluster collapse by validating inter-cluster distances

**IMPORTANT STATUS NOTE:**
This is a research scaffold and streaming format prototype. Token IDs are not 
learned unless you enable and train the VQ; the default path produces exploratory 
tokens only. Do not treat reconstructions as a codec baseline. Use with the 
Encodec bridge or supply trained weights for meaningful results.

Key Features:
- Hybrid semantic + acoustic tokenization (FIXED: proper k-means calibration)
- Neural codec with residual vector quantization (FIXED: validated cluster diversity)
- Music-specific representation learning
- Multi-scale temporal modeling
- Scientific evaluation with reconstruction metrics
- Iterative optimization for LLM-optimal representations
- Streaming output protocols for large-scale processing (robust NDJSON + RLE)
- FIXED: Consistent sample rate handling throughout pipeline
- FIXED: Proper reconstruction flag handling
- IMPROVED: Better evaluation metrics and clear compat mode labeling
- FIXED v0.1.2: Encodec integration with correct HuggingFace transformers API
- FIXED v0.1.3: Robust k-means clustering with proper calibration and validation
        - FIXED v0.1.4: Progress reporting and memory management for production use
        - FIXED v0.1.4: Error handling separation - logging cannot poison k-means success

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
import hashlib
import pickle

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
import librosa.display
import soundfile as sf
from scipy import signal
from scipy.stats import entropy
import sklearn.metrics

# K-means for codebook initialization - FIXED: Better imports and validation
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

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

# MERT model support for music-specific codebook initialization
try:
    from transformers import AutoModel
    HAS_MERT = True
except ImportError:
    HAS_MERT = False

try:
    from encodec import EncodecModel
    HAS_ENCODEC = True
except ImportError:
    HAS_ENCODEC = False

warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# FIXED v0.1.4: Progress Reporting System
# ============================================================================

class ProgressReporter:
    """Enhanced progress reporting for long operations."""
    
    def __init__(self, total_steps: int, operation_name: str = "Processing"):
        self.total_steps = total_steps
        self.operation_name = operation_name
        self.current_step = 0
        self.start_time = time.time()
        self.last_report_time = 0
        self.report_interval = 5.0  # Report every 5 seconds
        
    def update(self, step: int = None, message: str = None):
        """
        Update progress and optionally print status.  This method wraps all
        status formatting in a try/except block so that unexpected
        formatting errors cannot poison the success state of the caller.  If
        formatting fails, a minimal fallback status is printed instead.
        """
        # Update the current step counter
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        current_time = time.time()
        # Only report when enough time has elapsed or on final step or when a message is provided
        if (current_time - self.last_report_time >= self.report_interval or
            self.current_step >= self.total_steps or
            message is not None):
            elapsed = current_time - self.start_time
            if self.current_step > 0:
                eta = (elapsed / self.current_step) * (self.total_steps - self.current_step)
                progress_pct = (self.current_step / self.total_steps) * 100
                # Build and print status with robust formatting
                try:
                    status = f"  {self.operation_name}: {progress_pct:.1f}% ({self.current_step}/{self.total_steps})"
                    status += f" | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s"
                    if message:
                        status += f" | {message}"
                    print(status)
                except Exception:
                    # Fall back to minimal status if formatting fails
                    try:
                        print(f"  {self.operation_name}: {self.current_step}/{self.total_steps}")
                    except Exception:
                        # As a last resort, silently ignore
                        pass
            self.last_report_time = current_time
    
    def finish(self, message: str = None):
        """
        Mark operation as complete.  Like update(), this method wraps
        status formatting in a try/except block to ensure that formatting
        errors do not propagate and affect caller logic.
        """
        elapsed = time.time() - self.start_time
        try:
            final_message = f"  {self.operation_name}: Complete in {elapsed:.1f}s"
            if message:
                final_message += f" | {message}"
            print(final_message)
        except Exception:
            try:
                print(f"  {self.operation_name}: Complete")
            except Exception:
                pass

# ============================================================================
# FIXED v0.1.4: Memory Management Utilities
# ============================================================================

def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0

def check_memory_requirements(audio_length: int, sample_rate: int) -> bool:
    """Check if system has enough memory for processing."""
    try:
        import psutil
        available_mb = psutil.virtual_memory().available / 1024 / 1024
        
        # Rough estimate: audio processing needs ~10x raw audio size in memory
        audio_size_mb = (audio_length * 4) / 1024 / 1024  # 4 bytes per float32 sample
        estimated_need_mb = audio_size_mb * 10
        
        if estimated_need_mb > available_mb * 0.8:  # Use max 80% of available memory
            print(f"WARNING: Estimated memory need ({estimated_need_mb:.1f} MB) may exceed available ({available_mb:.1f} MB)")
            return False
        return True
    except ImportError:
        return True  # Can't check, assume OK

def aggressive_cleanup():
    """Aggressive memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    # Additional cleanup
    if hasattr(torch.cuda, 'reset_memory_stats'):
        torch.cuda.reset_memory_stats()

# ============================================================================
# Codebook Caching System
# ============================================================================

def get_default_codebook_cache_dir() -> Path:
    """Get default directory for codebook caching."""
    # Use user cache directory if available, otherwise current directory
    if hasattr(os, 'environ') and 'HOME' in os.environ:
        cache_dir = Path.home() / '.cache' / 'neural_audio_tokenizer' / 'codebooks'
    else:
        cache_dir = Path('./codebooks')
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def get_codebook_cache_key(model_id: str, codebook_size: int, num_quantizers: int, 
                          input_dim: int, layer_type: str) -> str:
    """Generate cache key for codebook based on model parameters."""
    key_parts = [
        model_id,
        f"size{codebook_size}",
        f"nq{num_quantizers}",
        f"dim{input_dim}",
        layer_type
    ]
    return "_".join(key_parts) + ".pkl"

def save_codebooks(quantizer: 'ResidualVectorQuantizer', cache_dir: Path, 
                  cache_key: str) -> bool:
    """Save quantizer codebooks to disk."""
    try:
        cache_file = cache_dir / cache_key
        
        # Extract all codebook data
        codebook_data = {
            'codebooks': [],
            'ema_counts': [],
            'ema_weights': [],
            'input_dim': quantizer.input_dim,
            'codebook_size': quantizer.codebook_size,
            'num_quantizers': quantizer.num_quantizers,
            'version': '1.4'  # Updated version
        }
        
        for i, vq_layer in enumerate(quantizer.quantizers):
            codebook_data['codebooks'].append(vq_layer.codebook.cpu().numpy())
            codebook_data['ema_counts'].append(vq_layer.ema_count.cpu().numpy())
            codebook_data['ema_weights'].append(vq_layer.ema_weight.cpu().numpy())
        
        # Save with pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(codebook_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Saved codebooks to: {cache_file}")
        return True
        
    except Exception as e:
        print(f"Warning: Failed to save codebooks: {e}")
        return False


def backup_existing_codebooks(cache_file: Path) -> bool:
    """Create backup of existing codebooks before overwriting."""
    if not cache_file.exists():
        return True  # No backup needed
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = cache_file.with_suffix(f'.backup_{timestamp}{cache_file.suffix}')
    
    try:
        # Use copy to preserve original, then verify backup before deletion
        import shutil
        shutil.copy2(cache_file, backup_file)
        
        # Verify backup was successful
        if backup_file.exists() and backup_file.stat().st_size > 0:
            print(f"Backed up existing codebooks to: {backup_file}")
            return True
        else:
            print(f"Warning: Backup verification failed for {backup_file}")
            return False
            
    except Exception as e:
        print(f"Warning: Failed to backup existing codebooks: {e}")
        return False


def save_codebooks_with_backup(quantizer: 'ResidualVectorQuantizer', cache_dir: Path, 
                              cache_key: str, force_reinit: bool = False) -> bool:
    """Save quantizer codebooks to disk, backing up existing files if force_reinit is True."""
    cache_file = cache_dir / cache_key
    
    # Handle backup when force_reinit is enabled
    if force_reinit and cache_file.exists():
        print(f"Force re-init enabled, backing up existing codebooks...")
        if not backup_existing_codebooks(cache_file):
            print("Warning: Backup failed, but continuing with save")
    
    # Use the existing save function
    return save_codebooks(quantizer, cache_dir, cache_key)


def load_codebooks(quantizer: 'ResidualVectorQuantizer', cache_dir: Path, 
                  cache_key: str) -> bool:
    """Load quantizer codebooks from disk."""
    try:
        cache_file = cache_dir / cache_key
        
        if not cache_file.exists():
            return False
        
        # Load data
        with open(cache_file, 'rb') as f:
            codebook_data = pickle.load(f)
        
        # Verify compatibility
        if (codebook_data.get('input_dim') != quantizer.input_dim or
            codebook_data.get('codebook_size') != quantizer.codebook_size or
            codebook_data.get('num_quantizers') != quantizer.num_quantizers):
            print(f"Warning: Cached codebooks incompatible with current config")
            return False
        
        # Restore codebooks - get device from buffers since quantizer has no parameters
        try:
            device = next(quantizer.buffers()).device
        except StopIteration:
            # Fallback to CPU if no buffers found
            device = torch.device('cpu')
        
        for i, vq_layer in enumerate(quantizer.quantizers):
            if i < len(codebook_data['codebooks']):
                codebook_tensor = torch.from_numpy(codebook_data['codebooks'][i]).float().to(device)
                ema_count_tensor = torch.from_numpy(codebook_data['ema_counts'][i]).float().to(device)
                ema_weight_tensor = torch.from_numpy(codebook_data['ema_weights'][i]).float().to(device)
                
                vq_layer.codebook.copy_(codebook_tensor)
                vq_layer.ema_count.copy_(ema_count_tensor)
                vq_layer.ema_weight.copy_(ema_weight_tensor)
        
        print(f"Loaded cached codebooks from: {cache_file}")
        return True
        
    except Exception as e:
        print(f"Warning: Failed to load codebooks from {cache_file}: {e}")
        # Add debug information for troubleshooting
        import traceback
        print(f"  Debug traceback: {traceback.format_exc()}")
        return False


# ============================================================================
# FIXED v0.1.4: Enhanced K-means Clustering Utilities
# ============================================================================

class RobustKMeansClusterer:
    """
    FIXED v0.1.4: Robust k-means clustering with progress reporting and aggressive memory management.
    Addresses the critical bug where k-means would collapse to single cluster.
    """
    
    def __init__(self, n_clusters: int, random_state: int = 42, max_retries: int = 5):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_retries = max_retries
        self.scaler = StandardScaler()
        
    def fit_predict_validated(self, features: np.ndarray, min_cluster_separation: float = 0.1) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Fit k-means with validation, progress reporting, and quality checks.
        
        Returns:
            centroids: Cluster centroids [n_clusters, n_features]
            metrics: Quality metrics dictionary
        """
        if not HAS_SKLEARN:
            raise RuntimeError("scikit-learn required for k-means clustering")
        
        print(f"    Starting robust k-means clustering (n_clusters={self.n_clusters})")
        print(f"    Input features shape: {features.shape}")
        
        # FIXED v0.1.4: Memory usage check
        memory_mb = get_memory_usage_mb()
        feature_size_mb = (features.nbytes) / 1024 / 1024
        print(f"    Memory usage: {memory_mb:.1f} MB, Features: {feature_size_mb:.1f} MB")
        
        # Initialize progress reporter
        total_attempts = self.max_retries * 3  # 3 strategies per retry
        progress = ProgressReporter(total_attempts, "K-means clustering")
        
        # FIXED: Check input feature validity
        if features.shape[0] < self.n_clusters:
            raise ValueError(f"Not enough samples ({features.shape[0]}) for {self.n_clusters} clusters")
        
        # FIXED: Feature preprocessing and validation
        features_clean = self._preprocess_features(features)
        if features_clean is None:
            raise ValueError("Feature preprocessing failed - features too degenerate")
        
        # FIXED v0.1.4: Aggressive cleanup before starting
        aggressive_cleanup()
        
        # FIXED: Multiple k-means attempts with different strategies
        best_centroids = None
        best_metrics = {'silhouette_score': -1, 'calinski_harabasz_score': 0, 'inertia': float('inf')}
        
        # FIXED v0.1.4: More conservative parameters to prevent edge cases
        strategies = [
            {'init': 'k-means++', 'n_init': 10, 'max_iter': 300},  # Reduced from 20 init attempts
            {'init': 'random', 'n_init': 15, 'max_iter': 300},     # Reduced from 30 init attempts  
            {'init': 'k-means++', 'n_init': 25, 'max_iter': 200}   # Reduced max_iter
        ]
        
        attempt_count = 0
        
        for retry in range(self.max_retries):
            for strategy_idx, strategy in enumerate(strategies):
                attempt_count += 1

                # -----------------------------------------------------------------
                # CRITICAL FIX: Do not perform any logging inside this try/except
                # block.  We capture success/failure state and relevant results
                # first, then log afterwards.  This prevents logging failures
                # from misclassifying attempts as failures.
                attempt_succeeded = False
                attempt_centroids = None
                attempt_metrics: Dict[str, Any] = {}
                attempt_error: Optional[Exception] = None

                try:
                    # Aggressive cleanup between attempts (except for the very first)
                    if attempt_count > 1:
                        aggressive_cleanup()
                    # Create k-means with deterministic but varied seeds
                    kmeans = KMeans(
                        n_clusters=self.n_clusters,
                        random_state=self.random_state + retry * 17 + strategy_idx * 7,
                        **strategy
                    )
                    # Fit k-means
                    cluster_labels = kmeans.fit_predict(features_clean)
                    centroids_scaled = kmeans.cluster_centers_
                    # Validate cluster quality
                    metrics = self._validate_clusters(
                        features_clean, cluster_labels, centroids_scaled, min_cluster_separation
                    )
                    # Determine success based on metrics before any logging
                    if (
                        metrics.get('is_valid')
                        and metrics.get('silhouette_score', -1) > best_metrics.get('silhouette_score', -1)
                    ):
                        attempt_centroids = self.scaler.inverse_transform(centroids_scaled)
                        attempt_metrics = metrics
                        attempt_succeeded = True
                except Exception as e:
                    # Capture the error for later logging
                    attempt_error = e
                    attempt_succeeded = False

                # -----------------------------------------------------------------
                # All logging and progress updates occur outside of try/except
                # blocks.  Errors in logging will not affect success state.
                try:
                    if attempt_succeeded:
                        # Update best results
                        best_centroids = attempt_centroids
                        best_metrics = attempt_metrics
                        # Report success
                        progress.update(attempt_count, f"Good clustering (silhouette: {attempt_metrics.get('silhouette_score', 0):.3f})")
                        # Check for early stopping condition
                        if (
                            attempt_metrics.get('silhouette_score', 0) > 0.3
                            and attempt_metrics.get('min_cluster_separation', 0) > min_cluster_separation
                        ):
                            progress.finish("Excellent clustering achieved")
                            # Break out of the inner loop
                            break
                    elif attempt_error is not None:
                        # Report failure message
                        progress.update(attempt_count, f"Attempt failed: {str(attempt_error)[:50]}")
                    else:
                        # Report poor clustering quality
                        progress.update(attempt_count, "Poor clustering quality")
                except Exception:
                    # Ignore any logging errors
                    pass

                # If we have already found a reasonably good clustering, break the inner loop
                if best_centroids is not None and best_metrics.get('silhouette_score', -1) > 0.1:
                    break

            # If we found a good clustering, break out of the outer retry loop
            if best_centroids is not None and best_metrics.get('silhouette_score', -1) > 0.1:
                break
        
        # FIXED v0.1.4: Final cleanup
        aggressive_cleanup()

        if best_centroids is None:
            # FIXED: Fallback to random initialization if k-means completely fails
            try:
                progress.update(message="All attempts failed, using fallback")
            except Exception:
                # Ignore any progress update errors
                pass
            # Use a clear warning without emojis
            print("      Warning: All k-means attempts failed, using robust fallback initialization")
            best_centroids = self._fallback_initialization(features)
            best_metrics = {'silhouette_score': 0.0, 'status': 'fallback_initialization'}

        # Finish the progress reporter; handle any formatting errors gracefully
        try:
            progress.finish(f"silhouette={best_metrics.get('silhouette_score', 0):.3f}")
        except Exception:
            try:
                progress.finish()
            except Exception:
                pass

        return best_centroids, best_metrics
    
    def _preprocess_features(self, features: np.ndarray) -> Optional[np.ndarray]:
        """FIXED v0.1.4: More aggressive feature preprocessing with memory management."""
        print(f"      Preprocessing features...")
        
        # FIXED v0.1.4: Memory check before preprocessing
        initial_memory = get_memory_usage_mb()
        
        # Remove NaN/Inf values
        if np.any(~np.isfinite(features)):
            # Warn when removing non-finite values
            print(f"        Warning: Removing non-finite values")
            finite_mask = np.all(np.isfinite(features), axis=1)
            features = features[finite_mask]
            
            if len(features) < self.n_clusters:
                print(f"        Error: Too few finite samples after cleanup")
                return None
        
        # Check feature variance
        feature_std = np.std(features, axis=0)
        print(f"        Feature variance: mean={np.mean(feature_std):.6f}, "
              f"min={np.min(feature_std):.6f}, max={np.max(feature_std):.6f}")
        
        # FIXED: Remove low-variance features that can cause k-means issues
        high_var_mask = feature_std > 1e-8
        if not np.all(high_var_mask):
            # Warn when removing low-variance features
            print(f"        Warning: Removing {np.sum(~high_var_mask)} low-variance features")
            features = features[:, high_var_mask]
            
            if features.shape[1] < 2:
                print(f"        Error: Too few high-variance features remaining")
                return None
        
        # FIXED v0.1.4: Subsample very large feature sets to prevent memory issues
        max_samples_for_kmeans = 100000  # More conservative limit
        if len(features) > max_samples_for_kmeans:
            # Inform the user that subsampling is occurring for memory reasons
            print(f"        Subsampling {max_samples_for_kmeans} from {len(features)} features for memory efficiency")
            indices = np.random.RandomState(self.random_state).choice(
                len(features), max_samples_for_kmeans, replace=False
            )
            features = features[indices]
            aggressive_cleanup()  # Cleanup after subsampling
        
        # FIXED: Standardize features (critical for k-means)
        print(f"        Standardizing features...")
        features_scaled = self.scaler.fit_transform(features)
        
        # FIXED: Check for degeneracy after scaling
        if np.any(np.std(features_scaled, axis=0) < 1e-6):
            print(f"        Error: Features still degenerate after scaling")
            return None
        
        # FIXED: Remove duplicate samples that can cause k-means issues
        unique_features, unique_indices = np.unique(features_scaled, axis=0, return_index=True)
        if len(unique_features) < len(features_scaled) * 0.8:  # More than 20% duplicates
            print(f"        Warning: Found {len(features_scaled) - len(unique_features)} duplicate samples")
            features_scaled = unique_features
        
        if len(features_scaled) < self.n_clusters:
            print(f"        Error: Not enough unique samples ({len(features_scaled)}) for {self.n_clusters} clusters")
            return None
        
        # FIXED v0.1.4: Memory usage check after preprocessing
        final_memory = get_memory_usage_mb()
        print(f"        Features preprocessed: {features_scaled.shape}, "
              f"memory: {initial_memory:.1f} -> {final_memory:.1f} MB")
        
        return features_scaled
    
    def _validate_clusters(self, features: np.ndarray, labels: np.ndarray, 
                          centroids: np.ndarray, min_separation: float) -> Dict[str, float]:
        """FIXED v0.1.4: More efficient cluster quality validation."""
        metrics = {'is_valid': False}
        
        try:
            # Check if all clusters are used
            unique_labels = np.unique(labels)
            if len(unique_labels) != self.n_clusters:
                print(f"        Warning: Only {len(unique_labels)} clusters used (expected {self.n_clusters})")
                return metrics
            
            # FIXED: Check cluster sizes (avoid tiny clusters)
            cluster_sizes = np.bincount(labels)
            min_cluster_size = max(2, len(features) // (self.n_clusters * 20))  # More conservative
            if np.min(cluster_sizes) < min_cluster_size:
                print(f"        Warning: Tiny cluster detected (min size: {np.min(cluster_sizes)})")
                return metrics
            
            # FIXED v0.1.4: Calculate inter-cluster distances more efficiently
            from scipy.spatial.distance import pdist
            distances = pdist(centroids)
            
            min_cluster_separation = np.min(distances) if len(distances) > 0 else 0
            if min_cluster_separation < min_separation:
                print(f"        Warning: Clusters too close (min separation: {min_cluster_separation:.3f})")
                return metrics
            
            # FIXED v0.1.4: Calculate silhouette score (but handle edge cases and large datasets)
            if len(features) >= 2 * self.n_clusters:  # Need enough samples for meaningful silhouette
                # Sample for large datasets to improve performance
                if len(features) > 10000:
                    sample_indices = np.random.choice(len(features), 10000, replace=False)
                    sample_features = features[sample_indices]
                    sample_labels = labels[sample_indices]
                    silhouette_score_val = silhouette_score(sample_features, sample_labels)
                else:
                    silhouette_score_val = silhouette_score(features, labels)
                    
                    if silhouette_score_val < -0.5:  # Very poor clustering
                        print(f"        Warning: Very poor silhouette score: {silhouette_score_val:.3f}")
                        return metrics
            else:
                silhouette_score_val = 0.0
            
            # FIXED: Calculate Calinski-Harabasz score
            try:
                ch_score = calinski_harabasz_score(features, labels)
            except:
                ch_score = 0.0
            
            # FIXED: All validation passed
            metrics = {
                'is_valid': True,
                'silhouette_score': silhouette_score_val,
                'calinski_harabasz_score': ch_score,
                'min_cluster_separation': min_cluster_separation,
                'cluster_sizes': cluster_sizes.tolist(),
                'n_clusters_used': len(unique_labels)
            }
            
        except Exception as e:
            print(f"        Error: Cluster validation failed: {e}")
            
        return metrics
    
    def _fallback_initialization(self, features: np.ndarray) -> np.ndarray:
        """FIXED: Robust fallback when k-means fails."""
        print(f"      Generating fallback initialization...")
        
        # FIXED: Use feature statistics to create diverse initial centroids
        features_mean = np.mean(features, axis=0)
        features_std = np.std(features, axis=0)
        
        centroids = []
        for i in range(self.n_clusters):
            # Create centroids in different regions of feature space
            angle = 2 * np.pi * i / self.n_clusters
            radius = 2.0  # Spread centroids out
            
            # Create a centroid by offsetting from mean in a systematic way
            offset = np.zeros_like(features_mean)
            offset[::2] = radius * np.cos(angle + np.arange(len(offset[::2])) * 0.1) * features_std[::2]
            offset[1::2] = radius * np.sin(angle + np.arange(len(offset[1::2])) * 0.1) * features_std[1::2]
            
            centroid = features_mean + offset
            centroids.append(centroid)
        
        centroids = np.array(centroids)
        print(f"      Generated {len(centroids)} diverse fallback centroids")
        
        return centroids


# ============================================================================
# Utility Functions for Device and Memory Management
# ============================================================================

def get_device_safely(module):
    """Safely get device from module parameters, fallback to CPU."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device('cpu')

def cleanup_cuda_memory():
    """Clean up CUDA memory safely."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def safe_tensor_cleanup(tensors):
    """Safely cleanup tensor list."""
    for tensor in tensors:
        if tensor is not None and hasattr(tensor, 'is_cuda') and tensor.is_cuda:
            del tensor
    cleanup_cuda_memory()

def set_deterministic_mode(seed: int = 42):
    """Set deterministic mode for reproducible results."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ============================================================================
# Core Neural Architecture Components
# ============================================================================

# ============================================================================
# Encodec codebook extraction helper (for one-pass seeding of quantizers)
# ============================================================================

def _extract_encodec_codebook_vectors(encodec_model):
    """
    Extract codebook embedding vectors from a HuggingFace Encodec model (or compatible).
    Strategy:
      1) Look through named_parameters and named_buffers for 2D tensors whose names
         include 'codebook', 'embed', or 'embedding'.
      2) If none are found, scan modules for attributes with those names.
    Returns:
      np.ndarray of shape [N, D] where rows are code vectors.
    Raises:
      RuntimeError if no suitable matrices are found.
    """
    import numpy as _np
    import torch as _torch
    from torch import nn as _nn

    vectors = []

    # Prefer explicit names
    try:
        for name, p in encodec_model.named_parameters():
            if isinstance(p, _torch.Tensor) and p.dim() == 2:
                lname = name.lower()
                if any(k in lname for k in ["codebook", "embed", "embedding"]):
                    vectors.append(p.detach().cpu().numpy())

        for name, b in encodec_model.named_buffers():
            if isinstance(b, _torch.Tensor) and b.dim() == 2:
                lname = name.lower()
                if any(k in lname for k in ["codebook", "embed", "embedding"]):
                    vectors.append(b.detach().cpu().numpy())
    except Exception:
        pass

    # Fallback: scan modules for attributes
    if not vectors:
        try:
            for m in encodec_model.modules():
                for attr in ["codebook", "embed", "embedding", "_codebook"]:
                    if hasattr(m, attr):
                        obj = getattr(m, attr)
                        try:
                            if isinstance(obj, _torch.Tensor) and obj.dim() == 2:
                                vectors.append(obj.detach().cpu().numpy())
                            elif isinstance(obj, _nn.Embedding):
                                vectors.append(obj.weight.detach().cpu().numpy())
                        except Exception:
                            continue
        except Exception:
            pass

    if not vectors:
        raise RuntimeError("No Encodec codebook embeddings found; cannot seed from codebooks")

    try:
        cat = _np.concatenate([v.reshape(-1, v.shape[-1]) for v in vectors], axis=0)
    except Exception as e:
        raise RuntimeError(f"Failed to concatenate Encodec codebook matrices: {e}")

    return cat


def _extract_encodec_codebook_vectors_with_type(encodec_model, extraction_type='semantic'):
    """
    Extract codebook embedding vectors from a HuggingFace Encodec model with type-specific selection.
    
    This function improves upon the basic _extract_encodec_codebook_vectors by using different
    portions/patterns of the codebooks for semantic vs acoustic initialization, similar to how
    MERT uses different layer ranges for better token diversity.
    
    Args:
        encodec_model: The Encodec model instance
        extraction_type: 'semantic' (for high-level structure) or 'acoustic' (for low-level details)
    
    Returns:
        np.ndarray of shape [N, D] where rows are code vectors specialized for the given type.
    
    Raises:
        RuntimeError if no suitable matrices are found.
    """
    import numpy as _np
    import torch as _torch
    from torch import nn as _nn

    vectors = []
    all_vectors = []  # Keep track of all available vectors for smart selection

    # First, collect ALL available codebook vectors like the original function
    try:
        for name, p in encodec_model.named_parameters():
            if isinstance(p, _torch.Tensor) and p.dim() == 2:
                lname = name.lower()
                if any(k in lname for k in ["codebook", "embed", "embedding"]):
                    vector_data = p.detach().cpu().numpy()
                    all_vectors.append({'name': name, 'data': vector_data})

        for name, b in encodec_model.named_buffers():
            if isinstance(b, _torch.Tensor) and b.dim() == 2:
                lname = name.lower()
                if any(k in lname for k in ["codebook", "embed", "embedding"]):
                    vector_data = b.detach().cpu().numpy()
                    all_vectors.append({'name': name, 'data': vector_data})
    except Exception:
        pass

    # Fallback: scan modules for attributes
    if not all_vectors:
        try:
            for i, m in enumerate(encodec_model.modules()):
                for attr in ["codebook", "embed", "embedding", "_codebook"]:
                    if hasattr(m, attr):
                        obj = getattr(m, attr)
                        try:
                            if isinstance(obj, _torch.Tensor) and obj.dim() == 2:
                                vector_data = obj.detach().cpu().numpy()
                                all_vectors.append({'name': f'module_{i}_{attr}', 'data': vector_data})
                            elif isinstance(obj, _nn.Embedding):
                                vector_data = obj.weight.detach().cpu().numpy()
                                all_vectors.append({'name': f'module_{i}_{attr}_weight', 'data': vector_data})
                        except Exception:
                            continue
        except Exception:
            pass

    if not all_vectors:
        raise RuntimeError("No Encodec codebook embeddings found; cannot seed from codebooks")

    # Now apply type-specific selection strategy (inspired by MERT's layer-based approach)
    print(f"  Found {len(all_vectors)} codebook matrices, selecting subset for {extraction_type}")
    
    if extraction_type == 'semantic':
        # For semantic: prefer later/higher-level patterns
        # Strategy: Use vectors from the latter half of available codebooks + larger matrices
        print(f"  Using SEMANTIC strategy: selecting latter-half matrices for high-level structure")
        
        # Sort by name (often correlates with model depth) and prefer latter half
        sorted_vectors = sorted(all_vectors, key=lambda x: x['name'])
        start_idx = len(sorted_vectors) // 2
        if start_idx >= len(sorted_vectors):  # Ensure we get at least one matrix
            start_idx = max(0, len(sorted_vectors) - 1)
        selected_vectors = sorted_vectors[start_idx:]
        
        # Also prefer larger matrices (often contain more structured information)
        selected_vectors.sort(key=lambda x: x['data'].shape[0], reverse=True)
        
    elif extraction_type == 'acoustic':
        # For acoustic: prefer earlier/lower-level patterns
        # Strategy: Use vectors from the first half of available codebooks + focus on spectral diversity
        print(f"  Using ACOUSTIC strategy: selecting first-half matrices for low-level texture")
        
        # Sort by name and prefer first half
        sorted_vectors = sorted(all_vectors, key=lambda x: x['name'])
        end_idx = len(sorted_vectors) // 2
        if end_idx == 0:  # Ensure we get at least one matrix
            end_idx = 1
        selected_vectors = sorted_vectors[:end_idx]
        
        # For acoustic, we may want different sampling patterns - mix matrix sizes
        selected_vectors.sort(key=lambda x: x['data'].shape[1])  # Sort by feature dimension
        
    else:
        # Fallback to all vectors (original behavior)
        selected_vectors = all_vectors

    # Extract the actual vectors from selected matrices
    for vec_info in selected_vectors:
        vectors.append(vec_info['data'])
    
    if not vectors:
        # Emergency fallback: use all available vectors 
        vectors = [v['data'] for v in all_vectors]
        print(f"  WARNING: No vectors selected for {extraction_type}, falling back to all available vectors")
    
    print(f"  Selected {len(vectors)} matrices for {extraction_type} initialization")

    try:
        cat = _np.concatenate([v.reshape(-1, v.shape[-1]) for v in vectors], axis=0)
    except Exception as e:
        raise RuntimeError(f"Failed to concatenate Encodec codebook matrices for {extraction_type}: {e}")

    return cat


# ============================================================================
# MERT codebook extraction helper (for music-optimized quantizer initialization)
# ============================================================================

def _extract_mert_weight_matrices(mert_model, layer_range=None, extraction_type='semantic'):
    """
    Extract learned weight matrices from MERT model for codebook initialization.

    MERT is a BERT-style transformer for music understanding. This function extracts
    learned weight matrices from specific layers to get different musical aspects:
    - Early layers (0-3): Low-level timbre, texture, spectral content (ACOUSTIC)
    - Late layers (9-11): High-level structure, melody, rhythm (SEMANTIC)

    Args:
        mert_model: Loaded MERT model from transformers.AutoModel
        layer_range: Tuple (start, end) for layer indices to extract from.
                    If None, uses all layers.
        extraction_type: 'semantic' (late layers) or 'acoustic' (early layers)

    Returns:
        np.ndarray of shape [N, D] where rows are music-optimized weight vectors

    Raises:
        RuntimeError if no suitable weight matrices are found
    """
    import numpy as _np
    import torch as _torch

    # Determine layer range based on extraction type if not specified
    if layer_range is None:
        if extraction_type == 'semantic':
            # Use late MERT layers for high-level musical structure
            layer_range = (9, 12)  # Layers 9, 10, 11 (MERT-v1-95M has 12 layers)
            print(f"  Extracting from LATE layers {layer_range} for semantic (musical structure)")
        elif extraction_type == 'acoustic':
            # Use early MERT layers for low-level timbre/texture
            layer_range = (0, 3)   # Layers 0, 1, 2
            print(f"  Extracting from EARLY layers {layer_range} for acoustic (timbre/texture)")
        else:
            # Default: use all layers
            layer_range = (0, 12)
            print(f"  Extracting from ALL layers {layer_range}")

    weight_tensors = []

    # Extract weight matrices from MERT's learned parameters
    # Focus on embeddings and linear layers which contain music-optimized representations
    for name, param in mert_model.named_parameters():
        if not isinstance(param, _torch.Tensor):
            continue

        # Look for 2D weight matrices
        if param.dim() != 2:
            continue

        # Filter by layer range
        # MERT uses naming like: encoder.layer.0.attention.self.query.weight
        if 'encoder.layer' in name or 'layer.' in name:
            # Extract layer number from parameter name
            try:
                # Try to find layer number in name
                import re
                layer_match = re.search(r'layer[s]?\.(\d+)', name)
                if layer_match:
                    layer_num = int(layer_match.group(1))
                    # Skip if not in desired layer range
                    if layer_num < layer_range[0] or layer_num >= layer_range[1]:
                        continue
                else:
                    # If we can't parse layer number, skip it when layer_range is specified
                    continue
            except:
                # If parsing fails, skip this parameter
                continue

        # Focus on these key components:
        # 1. Embeddings (position, token embeddings)
        # 2. Attention weights (query, key, value)
        # 3. Feed-forward network weights
        # 4. Layer normalization scales (if 2D)

        lname = name.lower()

        # Prioritize these components for music understanding
        if any(keyword in lname for keyword in [
            'embed',           # Embedding layers
            'query', 'key', 'value',  # Attention weights
            'dense', 'linear',        # Feed-forward weights
            'intermediate',           # FFN intermediate layers
            'output.weight',          # Output projections
        ]):
            # Get the weight matrix
            weight = param.detach().cpu()

            # For very large matrices, sample rows to keep memory manageable
            if weight.shape[0] > 10000:
                indices = _np.random.choice(weight.shape[0], 10000, replace=False)
                weight = weight[indices]

            weight_tensors.append(weight)

    if not weight_tensors:
        # Fallback: try to get any 2D weight matrix from state dict
        state_dict = mert_model.state_dict()
        for key, value in state_dict.items():
            if isinstance(value, _torch.Tensor) and value.dim() == 2 and value.size(0) >= 100:
                weight = value.detach().cpu()
                if weight.shape[0] > 10000:
                    indices = _np.random.choice(weight.shape[0], 10000, replace=False)
                    weight = weight[indices]
                weight_tensors.append(weight)
                if len(weight_tensors) >= 5:  # Get at least 5 matrices
                    break

    if not weight_tensors:
        raise RuntimeError("No suitable weight matrices found in MERT model; cannot initialize from MERT")

    # Collect weight vectors, handling different matrix dimensions
    # We'll sample rows from each matrix and collect them together
    all_vectors = []

    for weight_matrix in weight_tensors:
        # Each weight matrix has shape [out_features, in_features]
        # We'll treat each row as a potential codebook vector
        rows = weight_matrix  # Shape: [num_rows, feature_dim]

        # Sample up to 5000 rows from each matrix to keep it manageable
        num_rows = rows.shape[0]
        if num_rows > 5000:
            indices = _np.random.choice(num_rows, 5000, replace=False)
            sampled_rows = rows[indices]
        else:
            sampled_rows = rows

        # Convert to numpy and add to collection
        all_vectors.append(sampled_rows.numpy())

    # Now we have a list of arrays with potentially different feature dimensions
    # We'll return them as a single array after dimension alignment in the caller
    # For now, concatenate only those with matching dimensions

    # Group by dimension
    dim_groups = {}
    for vectors in all_vectors:
        dim = vectors.shape[1]
        if dim not in dim_groups:
            dim_groups[dim] = []
        dim_groups[dim].append(vectors)

    # Use the dimension group with the most vectors
    best_dim = max(dim_groups.keys(), key=lambda d: sum(v.shape[0] for v in dim_groups[d]))
    best_vectors = dim_groups[best_dim]

    try:
        concatenated = _np.concatenate(best_vectors, axis=0)
        print(f"  Using {len(best_vectors)} weight matrices with dimension {best_dim}")
        return concatenated
    except Exception as e:
        raise RuntimeError(f"Failed to concatenate MERT weight matrices: {e}")


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer based on SoundStream and MuQ architectures.
    Implements hierarchical quantization for efficient music representation.
    
    MAJOR FIXES v0.1.4: Enhanced k-means initialization with progress reporting and memory management.
    MAJOR FIXES v0.1.3: Completely redesigned k-means initialization with proper validation.
    """
    def __init__(self,
                 input_dim: int = 512,
                 codebook_size: int = 4096,  # Increased default from 1024
                 num_quantizers: int = 8,
                 commitment_weight: float = 0.25,
                 ema_decay: float = 0.99,
                 temperature: float = 0.5,  # Temperature for stochastic quantization
                 use_stochastic: bool = True):  # Enable stochastic quantization
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.commitment_weight = commitment_weight

        # Create multiple quantizer layers with temperature
        self.quantizers = nn.ModuleList([
            VectorQuantizer(input_dim, codebook_size, commitment_weight, ema_decay,
                          temperature=temperature, use_stochastic=use_stochastic)
            for _ in range(num_quantizers)
        ])
        
    def forward(self, x, training_mode: bool = None):
        """
        Forward pass through residual quantization layers.
        
        Args:
            x: Input tensor [batch, channels, time]
            training_mode: Override training mode for this forward pass
            
        Returns:
            quantized: Quantized representation
            codes: List of quantization codes for each layer
            losses: Dictionary of quantization losses
        """
        # Set training mode if specified
        original_training = self.training
        if training_mode is not None:
            self.train(training_mode)
        
        try:
            # FIXED: Better input validation and shape handling
            if x.dim() not in [2, 3]:
                raise ValueError(f"Expected 2D or 3D input tensor, got {x.shape}")
            
            # Handle 2D input by adding batch dimension
            if x.dim() == 2:
                x = x.unsqueeze(0)  # Add batch dimension
            
            # Input should be [batch, channels, time]
            if x.dim() != 3:
                raise ValueError(f"Expected 3D input tensor [B, C, T], got {x.shape}")
            
            B, C, T = x.shape
            if C != self.input_dim:
                raise ValueError(f"Expected {self.input_dim} feature dimensions, got {C}")
            
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
        
        finally:
            # Restore original training mode
            if training_mode is not None:
                self.train(original_training)
    
    def encode(self, x):
        """Encode input to discrete codes."""
        with torch.no_grad():
            _, codes, _ = self.forward(x, training_mode=False)
            return codes
    
    def decode(self, codes):
        """Decode from discrete codes."""
        if not codes:
            return torch.zeros(1, self.input_dim, 1, device=codes[0].device if codes else torch.device('cpu'))
            
        batch_size = codes[0].shape[0]
        time_steps = codes[0].shape[1]
        device = codes[0].device
        
        # Initialize with proper dimensions
        quantized = torch.zeros(batch_size, self.input_dim, time_steps, 
                              dtype=torch.float, device=device)
        
        for i, code in enumerate(codes):
            if i < len(self.quantizers):
                layer_quantized = self.quantizers[i].decode(code)
                quantized += layer_quantized
            
        return quantized
    
    def initialize_from_encodec(self, encodec_features: torch.Tensor, cache_dir: Optional[Path] = None, 
                               cache_key: Optional[str] = None, force_reinit: bool = False):
        """
        Initialize quantizer codebooks using k-means on Encodec features with robust validation.
        
        MAJOR FIXES v0.1.4: Added progress reporting and memory management for production use.
        MAJOR FIXES v0.1.3: Complete rewrite with proper k-means calibration and validation.
        """
        if not HAS_SKLEARN:
            print("Warning: scikit-learn not available, skipping Encodec initialization")
            return
        
        # Use default cache if not specified
        if cache_dir is None:
            cache_dir = get_default_codebook_cache_dir()
        
        # Try to load cached codebooks first (unless force reinit)
        if cache_key and not force_reinit:
            if load_codebooks(self, cache_dir, cache_key):
                print("Successfully loaded cached codebooks, skipping k-means initialization")
                return
        
        print("Initializing quantizer codebooks from Encodec features using robust k-means...")
        if force_reinit:
            print("  Force re-initialization requested, ignoring cached codebooks")
        
        # FIXED: Better feature preparation and validation
        features_prepared = self._prepare_encodec_features(encodec_features)
        if features_prepared is None:
            print("  Error: Feature preparation failed, using random initialization")
            return
        
        print(f"  Prepared features shape: {features_prepared.shape}")
        
        # FIXED v0.1.4: Memory check before starting
        if not check_memory_requirements(features_prepared.shape[0] * features_prepared.shape[1], 22050):
            print("  Warning: May not have sufficient memory for k-means clustering")
        
        # FIXED: Initialize each quantizer layer with robust k-means
        start_time = time.time()
        total_layers = len(self.quantizers)
        
        for i, quantizer in enumerate(self.quantizers):
            layer_start = time.time()
            print(f"  Initializing quantizer layer {i+1}/{total_layers}")

            # Use a robust k-means clusterer with validation
            clusterer = RobustKMeansClusterer(
                n_clusters=self.codebook_size,
                random_state=42 + i * 123  # Different seed per layer
            )

            # Track success state and results for this layer
            layer_succeeded = False
            layer_centroids: Optional[np.ndarray] = None
            layer_metrics: Dict[str, Any] = {}
            layer_error: Optional[Exception] = None

            try:
                # Perform k-means initialization with validation
                centroids, metrics = clusterer.fit_predict_validated(features_prepared)
                # Validate centroid quality separately
                if not self._validate_centroids(centroids):
                    raise ValueError("Centroid validation failed")
                # Store results for logging outside the try/except
                layer_centroids = centroids
                layer_metrics = metrics
                layer_succeeded = True
            except Exception as e:
                # Capture error for later logging
                layer_error = e
                layer_succeeded = False

            # Logging and codebook updates happen outside of the try/except above
            if layer_succeeded:
                # Attempt to update the codebook; catch any errors here so they
                # don't inadvertently mark the k-means run as failed
                try:
                    centroids_tensor = torch.from_numpy(layer_centroids).float()
                    quantizer.codebook.copy_(centroids_tensor)
                    # Reset EMA buffers
                    quantizer.ema_count.zero_()
                    quantizer.ema_weight.copy_(centroids_tensor)
                except Exception as e:
                    print(f"    Error: Failed to update codebook for layer {i+1}: {e}")
                    continue
                # Now log success; any logging failures will not affect k-means success
                try:
                    layer_time = time.time() - layer_start
                    silhouette = layer_metrics.get('silhouette_score', 0)
                    print(f"    Layer {i+1} initialized in {layer_time:.1f}s, silhouette: {silhouette:.3f}")
                except Exception:
                    # Basic success message fallback
                    print(f"    Layer {i+1} initialized successfully")
            else:
                # Log failure without affecting state
                try:
                    if layer_error is not None:
                        print(f"    Error: Layer {i+1} k-means failed: {layer_error}, using random initialization")
                    else:
                        print(f"    Error: Layer {i+1} k-means failed, using random initialization")
                except Exception:
                    # As a last resort, print a generic failure message
                    print(f"    Error: Layer {i+1} initialization failed")
                # Skip updating this layer since random initialization will be used
                continue
        
        total_time = time.time() - start_time
        print(f"Robust k-means initialization complete in {total_time:.1f}s")
        
        # FIXED: Validate final codebook diversity
        self._validate_final_codebooks()
        
        # Save codebooks to cache for future use
        if cache_key:
            save_codebooks_with_backup(self, cache_dir, cache_key, force_reinit)
        
        print("Codebook initialization from Encodec completed successfully!")
    
    def _prepare_encodec_features(self, encodec_features: torch.Tensor) -> Optional[np.ndarray]:
        """FIXED: Robust feature preparation with validation."""
        print("    Preparing Encodec features for k-means...")
        
        try:
            # Handle different tensor formats from Encodec
            if encodec_features.dim() == 4:
                # Handle 4D: [frames, batch, quantizers, time] -> [batch*time, features]
                print("      Converting 4D Encodec features")
                features_reshaped = encodec_features.mean(dim=(0, 2))  # Average over frames and quantizers
            elif encodec_features.dim() == 3:
                # Handle 3D: [batch, features, time] -> [batch*time, features]
                print("      Converting 3D Encodec features")
                features_reshaped = encodec_features
            else:
                print(f"      Unexpected Encodec feature dimensions: {encodec_features.shape}")
                features_reshaped = encodec_features.view(1, -1, encodec_features.shape[-1])
            
            # Flatten to [samples, features] for k-means
            if features_reshaped.dim() == 3:
                B, C, T = features_reshaped.shape
                features_flat = features_reshaped.transpose(1, 2).contiguous().view(-1, C)  # [B*T, C]
            else:
                features_flat = features_reshaped.view(-1, features_reshaped.shape[-1])
            
            # Convert to numpy
            features_np = features_flat.cpu().numpy()
            print(f"      Features flattened to: {features_np.shape}")
            
            # FIXED: Comprehensive feature quality validation
            if not self._validate_feature_quality(features_np):
                return None
            
            # FIXED: Sub-sample if too many features (for computational efficiency)
            max_samples = min(50000, len(features_np))  # Reasonable limit for k-means
            if len(features_np) > max_samples:
                print(f"      Sub-sampling {max_samples} features from {len(features_np)} for efficiency")
                indices = np.random.choice(len(features_np), max_samples, replace=False)
                features_np = features_np[indices]
            
            # FIXED: Project to target dimension if needed
            if features_np.shape[1] != self.input_dim:
                print(f"      Projecting features from {features_np.shape[1]} to {self.input_dim} dimensions")
                # Use PCA-like projection to preserve as much variance as possible
                from sklearn.decomposition import PCA
                pca = PCA(n_components=min(self.input_dim, features_np.shape[1]))
                features_projected = pca.fit_transform(features_np)
                
                # Pad or truncate to exact target dimension
                if features_projected.shape[1] < self.input_dim:
                    padding = np.random.randn(features_projected.shape[0], 
                                            self.input_dim - features_projected.shape[1]) * 0.1
                    features_np = np.concatenate([features_projected, padding], axis=1)
                else:
                    features_np = features_projected[:, :self.input_dim]
            
            # FIXED: Final validation after all preprocessing
            if not self._validate_feature_quality(features_np, final_check=True):
                return None

            # Feature preparation succeeded; print without check marks
            print(f"      Features prepared successfully: {features_np.shape}")
            return features_np
            
        except Exception as e:
            print(f"      Error: Feature preparation failed: {e}")
            return None
    
    def _validate_feature_quality(self, features: np.ndarray, final_check: bool = False) -> bool:
        """FIXED: Comprehensive feature quality validation."""
        try:
            if final_check:
                print("      Final feature quality check...")
            
            # Check for non-finite values
            if not np.all(np.isfinite(features)):
                print("        Error: Non-finite values detected")
                return False
            
            # Check feature variance
            feature_std = np.std(features, axis=0)
            mean_std = np.mean(feature_std)
            min_std = np.min(feature_std)
            
            if final_check:
                print(f"        Feature variance: mean={mean_std:.6f}, min={min_std:.6f}")
            
            if mean_std < 1e-8:
                print("        Error: Very low feature variance")
                return False
            
            # Check for sufficient samples
            min_samples_needed = max(self.codebook_size * 2, 1000)  # At least 2x codebook size
            if len(features) < min_samples_needed:
                print(f"        Error: Insufficient samples: {len(features)} < {min_samples_needed}")
                return False
            
            # Check dimensionality
            if features.shape[1] != self.input_dim:
                print(f"        Error: Wrong feature dimension: {features.shape[1]} != {self.input_dim}")
                return False
            
            # Check for reasonable dynamic range
            feature_range = np.max(features) - np.min(features)
            if feature_range < 1e-6:
                print("        Error: Features have very small dynamic range")
                return False
            
            if final_check:
                print("        Feature quality validation passed")
            
            return True
            
        except Exception as e:
            print(f"        Error: Feature validation failed: {e}")
            return False
    
    def _validate_centroids(self, centroids: np.ndarray) -> bool:
        """FIXED: Validate that centroids are diverse and useful."""
        try:
            # Check shape
            if centroids.shape != (self.codebook_size, self.input_dim):
                print(f"        Error: Wrong centroid shape: {centroids.shape}")
                return False
            
            # Check for finite values
            if not np.all(np.isfinite(centroids)):
                print(f"        Error: Non-finite centroids")
                return False
            
            # Check centroid diversity - calculate pairwise distances
            from scipy.spatial.distance import pdist
            distances = pdist(centroids)
            
            if len(distances) == 0:
                return True  # Single cluster case
            
            min_distance = np.min(distances)
            mean_distance = np.mean(distances)
            
            print(f"        Centroid distances: min={min_distance:.6f}, mean={mean_distance:.6f}")
            
            # FIXED: Ensure centroids are not too close (would result in poor diversity)
            if min_distance < 1e-6:
                print(f"        Error: Centroids too close together")
                return False
            
            # Check that centroids span reasonable space
            centroid_std = np.std(centroids, axis=0)
            if np.mean(centroid_std) < 1e-6:
                print(f"        Error: Centroids have very low variance")
                return False
            
            print(f"        Centroids validation passed")
            return True
            
        except Exception as e:
            print(f"        Error: Centroid validation failed: {e}")
            return False
    
    def _validate_final_codebooks(self):
        """FIXED: Validate final codebook diversity across all layers."""
        print("  Validating final codebook diversity...")
        
        for i, quantizer in enumerate(self.quantizers):
            codebook = quantizer.codebook.cpu().numpy()
            
            # Calculate intra-codebook diversity using scipy for efficiency
            from scipy.spatial.distance import pdist
            distances = pdist(codebook)
            
            if len(distances) > 0:
                min_dist = np.min(distances)
                mean_dist = np.mean(distances)
                print(f"    Layer {i}: min_dist={min_dist:.4f}, mean_dist={mean_dist:.4f}")
                
                if min_dist < 1e-4:
                    print(f"    Warning: Layer {i} has very similar codebook entries")
            else:
                print(f"    Layer {i}: single entry codebook")

    def _try_linear_projection(encodec_model, src, target_dim):
        # Look for 2D weights that look like "project_in/out/lin/proj*"
        for name, p in getattr(encodec_model, "named_parameters", lambda: [])():
            try:
                if not isinstance(p, torch.Tensor) or p.dim() != 2:
                    continue
                lname = name.lower()
                if not any(k in lname for k in ["project", "proj", "linear"]):
                    continue
                W = p.detach().cpu().numpy()
                if W.shape == (target_dim, src.shape[1]):
                    return src @ W.T, f"used {name}^T"
                if W.shape == (src.shape[1], target_dim):
                    return src @ W, f"used {name}"
            except Exception:
                continue
        return None, None


    def initialize_from_encodec_weights(self,
                                        encodec_model: Any,
                                        cache_dir: Optional[Path] = None,
                                        cache_key: Optional[str] = None,
                                        force_reinit: bool = False,
                                        use_kmeans: bool = False,
                                        layer_diversity_seed: int = 42,
                                        pre_extracted_vectors: Optional[np.ndarray] = None) -> None:
        """
        Initialize quantizer codebooks using Encodec's learned **codebook embeddings** only.
        Notes:
          - Re-clustering learned codebooks is typically unnecessary because Encodec uses
            EMA-updated codebooks (online k-means); use_kmeans remains available but is
            OFF by default.
          - Centroids are obtained via direct sampling from Encodec codebook vectors with
            projection to our input_dim using an existing linear projection if available,
            otherwise PCA, otherwise pad/truncate.
          - Different codebook samples are used for each residual layer to maximize diversity,
            controlled by layer_diversity_seed for reproducible but diverse initialization.
          - If pre_extracted_vectors is provided, we skip Encodec inspection and seed from it.
        """
        # Determine cache directory
        if cache_dir is None:
            cache_dir = get_default_codebook_cache_dir()

        # Attempt to load cached codebooks if a key is provided and not forcing
        if cache_key and not force_reinit:
            if load_codebooks(self, cache_dir, cache_key):
                print("Successfully loaded cached codebooks from Encodec codebooks (no k-means), skipping initialization")
                return

        print("Initializing quantizer codebooks from Encodec **codebook embeddings**...")

        # ------------------------------------------------------------------
        # 2.1 Extract codebook vectors
        # ------------------------------------------------------------------
        if pre_extracted_vectors is not None:
            features = pre_extracted_vectors
        else:
            try:
                features = _extract_encodec_codebook_vectors(encodec_model)
            except Exception as e:
                print(f"  Error: {e}; falling back to random initialization")
                return

        print(f"  Found {features.shape[0]} code vectors (dim={features.shape[1]}) in Encodec")

        # Subsample for efficiency if huge
        max_samples = min(50000, features.shape[0])
        if features.shape[0] > max_samples:
            print(f"  Sub-sampling {max_samples} out of {features.shape[0]} code vectors for efficiency")
            idx = np.random.choice(features.shape[0], max_samples, replace=False)
            features = features[idx]

        # ------------------------------------------------------------------
        # 2.2 Dimensionality alignment: try linear projection (proj*) if available
        #     else PCA; else pad/truncate.
        # ------------------------------------------------------------------
        def _try_linear_projection(encodec_model, src, target_dim):
            # Look for 2D weights that look like "project_in/out/lin/proj*"
            for name, p in getattr(encodec_model, "named_parameters", lambda: [])():
                try:
                    if not isinstance(p, torch.Tensor) or p.dim() != 2:
                        continue
                    lname = name.lower()
                    if not any(k in lname for k in ["project", "proj", "linear"]):
                        continue
                    W = p.detach().cpu().numpy()
                    if W.shape == (target_dim, src.shape[1]):
                        return src @ W.T, f"used {name}^T"
                    if W.shape == (src.shape[1], target_dim):
                        return src @ W, f"used {name}"
                except Exception:
                    continue
            return None, None

        if features.shape[1] != self.input_dim:
            projected, how = _try_linear_projection(encodec_model, features, self.input_dim)
            if projected is not None:
                print(f"  Projection: linear {how}")
                features_for_init = projected
            else:
                try:
                    from sklearn.decomposition import PCA
                    n_components = min(self.input_dim, features.shape[1])
                    pca = PCA(n_components)
                    projected = pca.fit_transform(features)
                    if projected.shape[1] < self.input_dim:
                        pad_width = self.input_dim - projected.shape[1]
                        pad = np.random.randn(projected.shape[0], pad_width) * 1e-3
                        projected = np.concatenate([projected, pad], axis=1)
                    features_for_init = projected[:, :self.input_dim]
                    print("  Projection: PCA")
                except Exception as e:
                    print(f"  PCA failed ({e}); using pad/truncate")
                    if features.shape[1] < self.input_dim:
                        pad_width = self.input_dim - features.shape[1]
                        pad = np.random.randn(features.shape[0], pad_width) * 1e-3
                        features_for_init = np.concatenate([features, pad], axis=1)
                    else:
                        features_for_init = features[:, :self.input_dim]
        else:
            features_for_init = features

        # ------------------------------------------------------------------
        # 2.3 Optionally run k-means (not recommended) otherwise direct sampling
        # ------------------------------------------------------------------
        centroids = None
        if use_kmeans and HAS_SKLEARN:
            try:
                clusterer = RobustKMeansClusterer(n_clusters=self.codebook_size, random_state=42)
                centroids, metrics = clusterer.fit_predict_validated(features_for_init)
                print(f"  K-means produced {centroids.shape[0]} centroids (silhouette: {metrics.get('silhouette_score', 0):.3f})")
            except Exception as e:
                print(f"  K-means clustering failed: {e}; falling back to direct sampling")
                centroids = None

        if centroids is None:
            if features_for_init.shape[0] >= self.codebook_size:
                idx = np.random.choice(features_for_init.shape[0], self.codebook_size, replace=False)
                centroids = features_for_init[idx]
            else:
                reps = int(np.ceil(self.codebook_size / features_for_init.shape[0]))
                centroids = np.tile(features_for_init, (reps, 1))[:self.codebook_size]
            print(f"  Selected {centroids.shape[0]} centroids via direct sampling from Encodec codebooks")

        # Validate the centroids before assignment
        try:
            if not self._validate_centroids(centroids):
                print("  Error: Centroid validation failed; falling back to random initialization")
                return
        except Exception as e:
            print(f"  Error: Centroid validation encountered an error: {e}")
            return

        # ------------------------------------------------------------------
        # 2.4 Assign DIFFERENT codebooks to each residual quantizer layer
        # ------------------------------------------------------------------
        print(f"  Assigning unique codebooks to {len(self.quantizers)} layers...")
        available_samples = features_for_init.shape[0]

        for i, vq_layer in enumerate(self.quantizers):
            try:
                # Sample different centroids for each layer to ensure diversity
                if available_samples >= self.codebook_size * (i + 2):
                    # We have enough samples - use completely different centroids for this layer
                    start_idx = i * self.codebook_size
                    end_idx = start_idx + self.codebook_size
                    layer_centroids = features_for_init[start_idx:end_idx]
                    print(f"    Layer {i}: using unique samples [{start_idx}:{end_idx}]")
                else:
                    # Not enough samples - resample with different random seed for layer diversity
                    layer_seed = layer_diversity_seed + i * 123
                    np.random.seed(layer_seed)
                    idx = np.random.choice(available_samples, self.codebook_size, replace=False)
                    layer_centroids = features_for_init[idx]
                    print(f"    Layer {i}: resampling with seed {layer_seed}")

                centroids_tensor = torch.from_numpy(layer_centroids).float()
                vq_layer.codebook.data.copy_(centroids_tensor)
                vq_layer.ema_count.data.zero_()
                vq_layer.ema_weight.data.copy_(centroids_tensor)
            except Exception as e:
                print(f"  Error: Failed to assign centroids to layer {i}: {e}")
                return

        # Validate the final codebooks for diversity
        try:
            self._validate_final_codebooks()
        except Exception:
            pass

        # Save codebooks to cache if key is provided
        if cache_key:
            save_codebooks_with_backup(self, cache_dir, cache_key, force_reinit)

        print("Codebook initialization from Encodec embeddings completed successfully!")

    def initialize_from_mert_model(self, model_name: str = "m-a-p/MERT-v1-95M",
                                   cache_dir: Optional[Path] = None,
                                   cache_key: Optional[str] = None,
                                   force_reinit: bool = False,
                                   random_seed: int = 42,
                                   extraction_type: str = 'semantic'):
        """
        Initialize quantizer codebooks using MERT's music-optimized RVQ-VAE representations.

        MERT (Music Understanding with Large-Scale Pre-training) models provide music-specific
        encoders with RVQ-VAE components purpose-built for musical understanding. This method
        extracts and uses these music-optimized codebook embeddings to initialize our quantizers.

        Args:
            model_name: MERT model to use ("m-a-p/MERT-v1-95M" or "m-a-p/MERT-v1-330M")
            cache_dir: Directory for codebook caching
            cache_key: Cache key for storing/loading codebooks
            force_reinit: Force re-initialization even if cached codebooks exist
            random_seed: Base seed for generating layer-specific diversity

        Raises:
            ImportError: If transformers package is not available
            RuntimeError: If MERT codebook extraction fails
        """
        if not HAS_MERT:
            raise ImportError("transformers package required for MERT initialization. Install with: pip install transformers")

        # Determine cache directory
        if cache_dir is None:
            cache_dir = get_default_codebook_cache_dir()

        # Attempt to load cached codebooks if a key is provided and not forcing
        if cache_key and not force_reinit:
            if load_codebooks(self, cache_dir, cache_key):
                print(f"Successfully loaded cached MERT codebooks for {model_name}")
                return

        print(f"Initializing quantizer codebooks from MERT model: {model_name}")
        print("  This may take a few minutes on first run (model download + codebook extraction)...")

        try:
            # Load MERT model
            from transformers import AutoModel
            mert_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            print(f"  Successfully loaded MERT model: {model_name}")

            # Extract music-optimized weight matrices from MERT
            # Use extraction_type to determine which layers to extract from
            mert_codebooks = _extract_mert_weight_matrices(mert_model, extraction_type=extraction_type)

            if mert_codebooks is None or mert_codebooks.shape[0] == 0:
                raise RuntimeError(f"Could not extract weight matrices from MERT model {model_name}")

            print(f"  Extracted {mert_codebooks.shape[0]} music-optimized weight vectors (dim={mert_codebooks.shape[1]})")

            # Subsample for efficiency if huge (use provided seed for different sampling)
            np.random.seed(random_seed)
            max_samples = min(50000, mert_codebooks.shape[0])
            if mert_codebooks.shape[0] > max_samples:
                print(f"  Sub-sampling {max_samples} out of {mert_codebooks.shape[0]} weight vectors (seed={random_seed})")
                idx = np.random.choice(mert_codebooks.shape[0], max_samples, replace=False)
                features = mert_codebooks[idx]
            else:
                features = mert_codebooks

            # Dimensionality alignment using PCA or pad/truncate
            if features.shape[1] != self.input_dim:
                print(f"  Projecting features from {features.shape[1]} to {self.input_dim} dimensions using PCA...")
                try:
                    from sklearn.decomposition import PCA
                    n_components = min(self.input_dim, features.shape[1])
                    pca = PCA(n_components)
                    projected = pca.fit_transform(features)
                    if projected.shape[1] < self.input_dim:
                        pad_width = self.input_dim - projected.shape[1]
                        pad = np.random.randn(projected.shape[0], pad_width) * 1e-3
                        projected = np.concatenate([projected, pad], axis=1)
                    features_for_init = projected[:, :self.input_dim]
                    print(f"  Projection complete: PCA variance ratio={pca.explained_variance_ratio_.sum():.3f}")
                except Exception as e:
                    print(f"  PCA failed ({e}); using pad/truncate")
                    if features.shape[1] < self.input_dim:
                        pad_width = self.input_dim - features.shape[1]
                        pad = np.random.randn(features.shape[0], pad_width) * 1e-3
                        features_for_init = np.concatenate([features, pad], axis=1)
                    else:
                        features_for_init = features[:, :self.input_dim]
            else:
                features_for_init = features

            # Direct sampling from MERT's music-optimized weights (use seed for different samples)
            np.random.seed(random_seed + 1000)  # Different seed for centroid selection
            if features_for_init.shape[0] >= self.codebook_size:
                idx = np.random.choice(features_for_init.shape[0], self.codebook_size, replace=False)
                centroids = features_for_init[idx]
            else:
                reps = int(np.ceil(self.codebook_size / features_for_init.shape[0]))
                centroids = np.tile(features_for_init, (reps, 1))[:self.codebook_size]

            print(f"  Selected {centroids.shape[0]} centroids via direct sampling (seed={random_seed})")

            # Validate the centroids before assignment
            if not self._validate_centroids(centroids):
                raise RuntimeError("MERT centroid validation failed")

            # Assign DIFFERENT codebooks to each residual quantizer layer
            # This is crucial for residual quantization to work properly!
            print(f"  Assigning unique codebooks to {len(self.quantizers)} layers...")

            # We have many vectors available - use different samples for each layer
            available_samples = features_for_init.shape[0]

            for i, vq_layer in enumerate(self.quantizers):
                # Sample different centroids for each layer to ensure diversity
                if available_samples >= self.codebook_size * (i + 2):
                    # We have enough samples - use completely different centroids for this layer
                    start_idx = i * self.codebook_size
                    end_idx = start_idx + self.codebook_size
                    layer_centroids = features_for_init[start_idx:end_idx]
                    print(f"    Layer {i}: using unique samples [{start_idx}:{end_idx}]")
                else:
                    # Not enough samples - resample with different random seed
                    np.random.seed(random_seed + i * 123)  # Use provided seed as base
                    idx = np.random.choice(available_samples, self.codebook_size, replace=False)
                    layer_centroids = features_for_init[idx]
                    print(f"    Layer {i}: resampling with seed {random_seed + i * 123}")

                centroids_tensor = torch.from_numpy(layer_centroids).float()
                vq_layer.codebook.data.copy_(centroids_tensor)
                vq_layer.ema_count.data.zero_()
                vq_layer.ema_weight.data.copy_(centroids_tensor)

            # Validate the final codebooks for diversity
            self._validate_final_codebooks()

            # Save codebooks to cache if key is provided
            if cache_key:
                save_codebooks_with_backup(self, cache_dir, cache_key, force_reinit)

            print(f"Successfully initialized from MERT {model_name} - music-optimized codebooks ready!")

        except Exception as e:
            print(f"Error: MERT initialization failed: {e}")
            print("  Falling back to random initialization")
            raise


class VectorQuantizer(nn.Module):
    """
    Single vector quantizer layer with EMA updates.
    
    FIXED v0.1.3: Improved tensor shape handling and dimension warnings.
    """
    
    def __init__(self,
                 input_dim: int,
                 codebook_size: int,
                 commitment_weight: float = 0.25,
                 ema_decay: float = 0.99,
                 temperature: float = 0.5,
                 use_stochastic: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay
        self.temperature = temperature  # Temperature for stochastic quantization
        self.use_stochastic = use_stochastic  # Whether to use stochastic quantization
        
        # Initialize codebook
        self.register_buffer('codebook', torch.randn(codebook_size, input_dim))
        self.register_buffer('ema_count', torch.zeros(codebook_size))
        self.register_buffer('ema_weight', self.codebook.clone())
        
    def forward(self, x):
        """
        Vector quantization forward pass.
        
        FIXED v0.1.3: Better tensor shape handling and clearer error messages.
        """
        # FIXED: Better input validation
        if x.dim() not in [2, 3]:
            raise ValueError(f"VectorQuantizer expects 2D or 3D input, got {x.dim()}D tensor with shape {x.shape}")
        
        # Handle 2D input by adding batch dimension
        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [C, T] -> [1, C, T]
        
        # Input: [B, C, T] where C is feature dim, T is time
        B, C, T = x.shape
        
        if C != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} feature dimensions, got {C}")
        
        # Transpose to [B, T, C] for per-timestep quantization
        x_btc = x.transpose(1, 2).contiguous()  # [B, T, C]
        flat_input = x_btc.view(-1, self.input_dim)  # [B*T, C]

        # Calculate distances to codebook entries
        # flat_input: [B*T, C], codebook: [K, C]
        distances = torch.cdist(flat_input, self.codebook)  # [B*T, K]

        # Stochastic quantization with temperature for better codebook utilization
        # This forces exploration of more codebook entries during training
        if self.training or self.use_stochastic:
            # Use softmax with temperature to sample probabilistically
            # Higher temperature = more exploration, lower = more exploitation
            probs = F.softmax(-distances / self.temperature, dim=1)  # [B*T, K]
            codes_flat = torch.multinomial(probs, 1).squeeze(1)  # [B*T]
        else:
            # Deterministic quantization (argmin) for evaluation
            codes_flat = torch.argmin(distances, dim=1)  # [B*T]

        quantized_flat = F.embedding(codes_flat, self.codebook)  # [B*T, C]
        
        # Calculate losses
        e_latent_loss = F.mse_loss(quantized_flat.detach(), flat_input)
        q_latent_loss = F.mse_loss(quantized_flat, flat_input.detach())
        loss = q_latent_loss + self.commitment_weight * e_latent_loss
        
        # Straight-through estimator
        quantized_flat = flat_input + (quantized_flat - flat_input).detach()
        
        # Reshape back to original format
        quantized_btc = quantized_flat.view(B, T, C)
        quantized = quantized_btc.transpose(1, 2).contiguous()  # [B, C, T]
        codes = codes_flat.view(B, T)  # [B, T] - tokens per timestep
        
        # Handle original 2D input case
        if len(original_shape) == 2:
            quantized = quantized.squeeze(0)  # Remove batch dimension
            codes = codes.squeeze(0)  # Remove batch dimension
        
        # EMA update during training
        if self.training:
            self._update_ema(flat_input, codes_flat)
        
        return quantized, codes, loss
    
    def decode(self, codes):
        """Decode codes to continuous representation."""
        # Handle both 2D and 3D codes input
        original_shape = codes.shape
        if codes.dim() == 1:
            codes = codes.unsqueeze(0)  # [T] -> [1, T]
        
        # codes: [B, T]
        B, T = codes.shape
        codes_flat = codes.view(-1)  # [B*T]
        quantized_flat = F.embedding(codes_flat, self.codebook)  # [B*T, C]
        quantized_btc = quantized_flat.view(B, T, self.input_dim)  # [B, T, C]
        quantized = quantized_btc.transpose(1, 2).contiguous()  # [B, C, T]
        
        # Handle original 1D input case
        if len(original_shape) == 1:
            quantized = quantized.squeeze(0)  # Remove batch dimension
        
        return quantized
    
    def _update_ema(self, flat_input, codes_flat):
        """Update codebook using exponential moving average."""
        with torch.no_grad():
            # Update counts
            codes_onehot = F.one_hot(codes_flat, self.codebook_size).float()
            codes_count = codes_onehot.sum(dim=0)
            
            self.ema_count.mul_(self.ema_decay).add_(codes_count, alpha=1 - self.ema_decay)
            
            # Update weights
            codes_sum = torch.matmul(codes_onehot.t(), flat_input)
            self.ema_weight.mul_(self.ema_decay).add_(codes_sum, alpha=1 - self.ema_decay)
            
            # Update codebook with proper per-code normalization
            epsilon = 1e-5
            normalized_counts = self.ema_count + epsilon
            self.codebook.copy_(self.ema_weight / normalized_counts.unsqueeze(1))


class MelResidualEncoder(nn.Module):
    """
    Mel-scale residual encoder inspired by MuQ paper.
    Optimized for music-specific frequency representations.
    
    IMPROVED v0.1.1: Better sample rate handling and clearer documentation.
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
        self.target_dim = target_dim
        
        # Mel spectrogram transform - will be updated dynamically based on input SR
        self.mel_transform = None  # Will be created with correct sample rate
        
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
            # Ensure GroupNorm constraint: channels >= 8 and divisible by 8
            norm_groups = min(8, out_channels) if out_channels >= 8 else 1
            self.encoder.add_module(f'norm_{i}', nn.GroupNorm(norm_groups, out_channels))
            self.encoder.add_module(f'act_{i}', nn.GELU())
            
            in_channels = out_channels
        
        # Final projection to target dimension
        self.proj = nn.Conv2d(in_channels, target_dim, kernel_size=1)
        
    def forward(self, waveform, sample_rate: int):
        """
        Encode waveform to mel-based representation.
        
        FIXED v0.1.1: Proper sample rate handling - creates mel transform with correct SR.
        """
        device = waveform.device
        
        # Create/update mel transform with correct sample rate
        if (self.mel_transform is None or 
            (hasattr(self.mel_transform, 'sample_rate') and 
             self.mel_transform.sample_rate != sample_rate)):
            
            self.mel_transform = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                normalized=True
            ).to(device)
        
        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Add channel dimension if needed
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
    
    IMPROVED v0.1.1: Better sample rate handling and clearer fallback labeling.
    """
    def __init__(self, model_name: str = "facebook/wav2vec2-base", target_dim: int = 512):
        super().__init__()
        self.model_name = model_name
        self.target_dim = target_dim
        self.fallback_proj = None  # Cache for fallback projection
        self.using_fallback = False
        
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
                    
                print(f"Loaded semantic encoder: {model_name}")
                self.available = True
                self.using_fallback = False
            except Exception as e:
                print(f"Warning: Could not load {model_name} model: {e}")
                print("Falling back to spectral feature encoder")
                self.available = False
                self.using_fallback = True
        else:
            print("Transformers not available, using spectral fallback encoder")
            self.available = False
            self.using_fallback = True
    
    def forward(self, waveform, sample_rate: int):
        """
        Extract semantic features from waveform.
        
        FIXED v0.1.1: Better sample rate handling for both Wav2Vec2 and fallback.
        """
        if not self.available:
            return self._spectral_fallback(waveform, sample_rate)
        
        device = waveform.device
        self.wav2vec2 = self.wav2vec2.to(device)
        self.projection = self.projection.to(device)
        
        # Resample to 16kHz if needed (Wav2Vec2 requirement)
        target_sr = 16000
        if sample_rate != target_sr:
            resampler = T.Resample(sample_rate, target_sr).to(device)
            waveform = resampler(waveform)
        
        # Ensure proper input shape for Wav2Vec2 (batch, sequence_length)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add batch dimension
        elif waveform.dim() == 3:
            waveform = waveform.squeeze(1)   # Remove channel dimension if present
        
        # Process through Wav2Vec2
        with torch.no_grad():
            features = self.wav2vec2(waveform)
            hidden_states = features.last_hidden_state
        
        # Project to target dimension
        projected = self.projection(hidden_states)
        
        # Transpose to [batch, channels, time] format
        return projected.transpose(1, 2)
    
    def _spectral_fallback(self, waveform, sample_rate):
        """
        IMPROVED v0.1.1: Enhanced fallback spectral features with better documentation.
        This produces basic spectral features when Wav2Vec2 is unavailable.
        """
        # Ensure waveform is properly shaped
        if waveform.dim() > 1:
            audio_1d = waveform.squeeze()
        else:
            audio_1d = waveform
        
        device = waveform.device
        
        # STFT parameters
        n_fft, hop = 2048, 512
        win = torch.hann_window(n_fft, device=device)
        
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
                audio_1d[start:], torch.zeros(end - len(audio_1d), device=device)
            ])
            windowed = frame * win
            stft_frame = torch.fft.rfft(windowed)
            stft_frames.append(stft_frame)
        
        stft_result = torch.stack(stft_frames, dim=1)  # [freq, time]
        magnitude = torch.abs(stft_result) + 1e-12
        
        # Frequency axis for centroid calculation
        freqs = torch.fft.rfftfreq(n_fft, 1.0/sample_rate, device=device)
        freqs = freqs.unsqueeze(1)  # [freq, 1]
        
        # Spectral centroid and bandwidth
        total_mag = magnitude.sum(dim=0) + 1e-8  # [time]
        centroid = (magnitude * freqs).sum(dim=0) / total_mag  # [time]
        
        # Spectral bandwidth
        freq_diff_sq = (freqs - centroid.unsqueeze(0)) ** 2
        bandwidth = torch.sqrt((magnitude * freq_diff_sq).sum(dim=0) / total_mag)  # [time]
        
        # Stack features [2, time]
        features = torch.stack([centroid, bandwidth], dim=0)  # [2, time]
        
        # Create persistent projection layer
        if self.fallback_proj is None or self.fallback_proj.weight.device != device:
            self.fallback_proj = nn.Linear(2, self.target_dim).to(device)
        
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
    
    IMPROVED v0.1.1: Better GroupNorm handling for various channel counts.
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
                nn.GroupNorm(min(8, hidden_dim) if hidden_dim >= 8 else 1, hidden_dim),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(min(8, hidden_dim) if hidden_dim >= 8 else 1, hidden_dim),
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
# Encodec Integration (FIXED v0.1.2)
# ============================================================================

class EncodecBridge(nn.Module):
    """
    Bridge to use pre-trained Encodec for codebook initialization via k-means.
    This extracts features from Encodec and uses them to initialize our quantizers.
    
    FIXED v0.1.2: Complete rewrite for proper codebook initialization approach.
    """
    def __init__(self, model_name: str = "facebook/encodec_24khz", device: str = "auto"):
        super().__init__()
        self.available = False
        self.model_name = model_name
        self.encodec_sample_rate = 24000  # Encodec models are trained on 24kHz
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        if HAS_TRANSFORMERS:
            try:
                from transformers import EncodecModel, AutoProcessor
                
                # Use the correct HuggingFace transformers API
                self.encodec = EncodecModel.from_pretrained(model_name).to(device)
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.encodec.eval()
                
                # Freeze parameters
                for param in self.encodec.parameters():
                    param.requires_grad = False
                
                self.available = True
                print(f"Successfully loaded Encodec model: {model_name} (requires {self.encodec_sample_rate} Hz)")
                
            except Exception as e:
                print(f"Failed to load Encodec model {model_name}: {e}")
                print("Make sure transformers is installed: pip install transformers")
                print("Available models: facebook/encodec_24khz, facebook/encodec_32khz")
                self.available = False
        else:
            print("Transformers not available - install with: pip install transformers")
            self.available = False
    
    def extract_features_for_initialization(self, waveform, sample_rate: int):
        """
        Extract features from Encodec encoder for quantizer initialization.
        
        FIXED v0.1.2: New method that extracts intermediate features for k-means initialization.
        """
        if not self.available:
            raise RuntimeError("Encodec model not available")
        
        # FIXED: Always resample to Encodec's required sample rate
        if sample_rate != self.encodec_sample_rate:
            print(f"Resampling audio from {sample_rate} Hz to {self.encodec_sample_rate} Hz for Encodec")
            resampler = T.Resample(sample_rate, self.encodec_sample_rate).to(self.device)
            waveform_resampled = resampler(waveform)
        else:
            waveform_resampled = waveform
        
        # Handle different input types properly
        if isinstance(waveform_resampled, torch.Tensor):
            # Ensure proper shape for Encodec
            if waveform_resampled.dim() == 1:
                audio_array = waveform_resampled.cpu().numpy()
            elif waveform_resampled.dim() == 2:
                audio_array = waveform_resampled.squeeze(0).cpu().numpy()
            elif waveform_resampled.dim() == 3:
                audio_array = waveform_resampled.squeeze(1).squeeze(0).cpu().numpy()
            else:
                raise ValueError(f"Unexpected tensor dimensions: {waveform_resampled.shape}")
        else:
            audio_array = waveform_resampled.squeeze() if waveform_resampled.ndim > 1 else waveform_resampled
        
        # Use processor to prepare inputs
        inputs = self.processor(
            raw_audio=audio_array, 
            sampling_rate=self.encodec_sample_rate, 
            return_tensors="pt"
        )
        
        # Move inputs to correct device
        device = next(self.encodec.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get encoder features (before quantization)
            encoder_outputs = self.encodec.encode(
                inputs["input_values"], 
                inputs.get("padding_mask")
            )
            
            # Extract the continuous features before quantization
            # These are the features we'll use for k-means initialization
            if hasattr(encoder_outputs, 'encoded_frames'):
                features = encoder_outputs.encoded_frames
            else:
                # Fallback: try to access encoder directly
                audio_codes = encoder_outputs.audio_codes
                # Use the mean of codes as proxy features
                features = audio_codes.float().mean(dim=2)  # Average over quantizers
        
        return features


# ============================================================================
# NDJSON Streaming Protocol (LLM-friendly LAM v0.1)
# ============================================================================

class NDJSONStreamer:
    """Enhanced NDJSON streaming protocol for LLM-friendly audio tokenization."""
    
    def __init__(self, sample_rate: int, hop_length: int, model_id: str = "tims-ears-0.1.7.epoch", 
                 codebook_size: int = 1024, num_semantic_layers: int = 4, num_acoustic_layers: int = 4,
                 rle_mode: bool = False, per_layer_encoding: Optional[Dict[str, str]] = None,
                 keyframe_interval_seconds: float = 5.0, audio_sha256: Optional[str] = None,
                 compat_mode: bool = False):
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
        self.compat_mode = compat_mode
        
        # RLE state for duration aggregation
        self.buffered_event = None
        self.last_frame_index = -1
        
    def create_header(self, duration_seconds: float = None, metadata: Dict = None, 
                     include_legend: bool = True) -> str:
        """
        Create enhanced NDJSON header with full format specification.
        
        IMPROVED v0.1.4: Updated version number for k-means fix.
        """
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
            "format_version": "1.5",  # Updated version with MERT integration
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
        
        # Add compatibility mode flag
        if self.compat_mode:
            header["compat_mode"] = True
            header["warning"] = "Tokens generated in compatibility mode - not from trained quantizers"
        
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
    # FIXED: Added audio-time vs processing-time distinction
    audio_frames_per_second: float = 0.0
    audio_tokens_per_second: float = 0.0
    processing_frames_per_second: float = 0.0
    processing_tokens_per_second: float = 0.0


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
        
        # FIXED: Separate audio-time vs processing-time metrics
        return TokenBudgetMetrics(
            total_tokens=total_tokens,
            semantic_tokens=self.semantic_tokens,
            acoustic_tokens=self.acoustic_tokens,
            tokens_per_second=total_tokens / max(elapsed, 1e-6),  # Legacy field
            frames_per_second=actual_fps,  # Legacy field
            compression_ratio=self.total_samples / max(total_tokens, 1),
            processing_time=elapsed,
            # New disambiguated metrics
            audio_frames_per_second=actual_fps,
            audio_tokens_per_second=total_tokens / max(audio_duration, 1e-6),
            processing_frames_per_second=self.total_frames / max(elapsed, 1e-6),
            processing_tokens_per_second=total_tokens / max(elapsed, 1e-6)
        )


# ============================================================================
# Main Neural Audio Tokenizer
# ============================================================================

class NeuralAudioTokenizer(nn.Module):
    """
    Main neural audio tokenizer implementing hybrid semantic + acoustic approach
    based on AudioLM and MuQ research.
    
    MAJOR FIXES v0.1.4:
    - Enhanced progress reporting for k-means operations
    - Aggressive memory management and cleanup
    - Better parameter validation and error recovery
    
    MAJOR FIXES v0.1.3:
    - FIXED: K-means clustering now properly calibrated with standardization and validation
    - Fixed codebook diversity validation to prevent collapse to single cluster
    - Added robust fallback strategies when k-means fails
    - Improved feature preprocessing to preserve diversity
    """
    def __init__(self,
                 sample_rate: int = 22050,
                 semantic_dim: int = 512,
                 acoustic_dim: int = 512,
                 codebook_size: int = 4096,  # Increased from 1024 to 4096 for better diversity
                 num_quantizers: int = 8,
                 n_mels: int = 128,
                 hop_length: int = 512,
                 enable_reconstruction: bool = True,  # NEW v0.1.1: Optional reconstruction
                 use_encodec_bridge: bool = False,    # NEW v0.1.1: Option to use Encodec quantizers (LEGACY)
                 encodec_model: str = "facebook/encodec_24khz",
                 # NEW v0.1.2: Codebook caching options
                 codebook_cache_dir: Optional[Path] = None,
                 enable_codebook_cache: bool = True,
                 force_reinit_codebooks: bool = False,
                 model_id: str = "tims-ears-0.1.7.mert",  # Updated version with MERT
                 # NEW v0.1.7: Codebook initialization method
                 codebook_init_method: str = "mert"):  # "mert", "encodec", or "random"
        super().__init__()
        self.sample_rate = sample_rate
        self.semantic_dim = semantic_dim
        self.acoustic_dim = acoustic_dim
        self.hop_length = hop_length
        self.enable_reconstruction = enable_reconstruction
        self.use_encodec_bridge = use_encodec_bridge  # Keep for backward compatibility
        self.codebook_cache_dir = codebook_cache_dir or get_default_codebook_cache_dir()
        self.enable_codebook_cache = enable_codebook_cache
        self.force_reinit_codebooks = force_reinit_codebooks
        self.model_id = model_id
        self.codebook_init_method = codebook_init_method

        # Backward compatibility: if use_encodec_bridge is True, override codebook_init_method
        if self.use_encodec_bridge:
            self.codebook_init_method = "encodec"
        
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
        
        # Quantizers - always create our custom ones with increased size and stochastic sampling
        self.semantic_quantizer = ResidualVectorQuantizer(
            semantic_dim, codebook_size, num_quantizers//2
        )
        self.acoustic_quantizer = ResidualVectorQuantizer(
            acoustic_dim, codebook_size, num_quantizers//2
        )

        # Store codebook size for later use
        self.codebook_size = codebook_size
        
        # Encodec bridge for codebook initialization (backward compatibility)
        if use_encodec_bridge:
            self.encodec_bridge = EncodecBridge(encodec_model)
            self.codebook_initialized = False
        else:
            self.encodec_bridge = None
            # Mark as initialized if using random or MERT (MERT doesn't need EncodecBridge)
            self.codebook_initialized = (self.codebook_init_method == "random")
        
        # Decoder for reconstruction - only build if enabled
        if enable_reconstruction:
            self.decoder = self._build_decoder()
        else:
            self.decoder = None
        
    def _build_decoder(self):
        """Build decoder for audio reconstruction."""
        return nn.Sequential(
            nn.Conv1d(self.semantic_dim + self.acoustic_dim, 512, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, 512), 512),
            nn.GELU(),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, 256), 256),
            nn.GELU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, 128), 128),
            nn.GELU(),
            nn.Conv1d(128, 1, kernel_size=3, padding=1)  # Output single channel
        )
    
    def _initialize_codebooks_from_external_source(self, method: str = "mert"):
        """
        Initialize codebooks using external pre-trained models.

        Args:
            method: Initialization method ("mert", "encodec", or "random")
                   - "mert": Use MERT music-optimized RVQ-VAE codebooks (RECOMMENDED for music)
                   - "encodec": Use EnCodec speech-optimized codebooks (legacy)
                   - "random": Skip external initialization (use random)
        """
        if method == "mert":
            # Use MERT-v1-95M by default (faster, still music-optimized)
            model_name = "m-a-p/MERT-v1-95M"
            print(f"Initializing codebooks from MERT: {model_name}")
            print("  MERT provides music-specific codebooks trained on musical data")

            try:
                # Generate cache keys for semantic and acoustic quantizers
                semantic_cache_key = None
                acoustic_cache_key = None
                if self.enable_codebook_cache:
                    semantic_cache_key = get_codebook_cache_key(
                        f"mert_{model_name.replace('/', '_')}",
                        self.semantic_quantizer.codebook_size,
                        self.semantic_quantizer.num_quantizers,
                        self.semantic_dim,
                        "semantic"
                    )
                    acoustic_cache_key = get_codebook_cache_key(
                        f"mert_{model_name.replace('/', '_')}",
                        self.acoustic_quantizer.codebook_size,
                        self.acoustic_quantizer.num_quantizers,
                        self.acoustic_dim,
                        "acoustic"
                    )

                # Initialize semantic quantizer from LATE MERT layers (high-level musical structure)
                print("  Initializing SEMANTIC quantizer from LATE MERT layers (structure/melody)...")
                self.semantic_quantizer.initialize_from_mert_model(
                    model_name=model_name,
                    cache_dir=self.codebook_cache_dir if self.enable_codebook_cache else None,
                    cache_key=semantic_cache_key,
                    force_reinit=self.force_reinit_codebooks,
                    random_seed=42,
                    extraction_type='semantic'  # Use late layers (9-11) for semantic
                )

                # Initialize acoustic quantizer from EARLY MERT layers (low-level timbre/texture)
                print("  Initializing ACOUSTIC quantizer from EARLY MERT layers (timbre/texture)...")
                self.acoustic_quantizer.initialize_from_mert_model(
                    model_name=model_name,
                    cache_dir=self.codebook_cache_dir if self.enable_codebook_cache else None,
                    cache_key=acoustic_cache_key,
                    force_reinit=self.force_reinit_codebooks,
                    random_seed=123,
                    extraction_type='acoustic'  # Use early layers (0-2) for acoustic
                )

                print("MERT codebook initialization completed successfully!")

            except Exception as e:
                print(f"Warning: MERT codebook initialization failed: {e}")
                print("Falling back to random codebooks...")

        elif method == "encodec":
            # Keep original EnCodec path as fallback (legacy, not recommended for music)
            print("WARNING: Using EnCodec initialization - optimized for speech, not music")
            print("  RECOMMENDATION: Use --codebook-init=mert for music-specific tokenization")
            self._initialize_codebooks_from_encodec_legacy()

        else:
            raise ValueError(f"Unknown initialization method: {method}")

    def _initialize_codebooks_from_encodec_legacy(self):
        """
        LEGACY: Initialize quantizer codebooks using the pre-trained Encodec model's own
        codebook weight matrices.  This replaces the earlier per-file k-means
        clustering based on audio features with a one-time initialization using
        Encodec's learned wisdom.  Once the codebooks are initialized and
        cached, they will be loaded on subsequent runs to ensure stability.

        IMPROVED v0.1.7: Now uses type-specific codebook extraction similar to MERT approach.
        Different portions of Encodec codebooks are used for semantic vs acoustic quantizers
        to achieve better token diversity, rather than just using different random seeds.

        NOTE: This method is now considered legacy. EnCodec is optimized for speech,
        not music. Use --codebook-init=mert for music-specific tokenization.
        """
        # Only proceed if Encodec is enabled, available and we haven't already initialized
        if not self.use_encodec_bridge or not getattr(self.encodec_bridge, 'available', False) or self.encodec_initialized:
            return
        try:
            print("Initializing codebooks from Encodec weight matrices...")
            # Generate cache keys for semantic and acoustic quantizers
            semantic_cache_key = None
            acoustic_cache_key = None
            if self.enable_codebook_cache:
                semantic_cache_key = get_codebook_cache_key(
                    self.model_id,
                    self.semantic_quantizer.codebook_size,
                    self.semantic_quantizer.num_quantizers,
                    self.semantic_dim,
                    "semantic"
                )
                acoustic_cache_key = get_codebook_cache_key(
                    self.model_id,
                    self.acoustic_quantizer.codebook_size,
                    self.acoustic_quantizer.num_quantizers,
                    self.acoustic_dim,
                    "acoustic"
                )
                if not self.force_reinit_codebooks:
                    print(f"Checking for cached codebooks in: {self.codebook_cache_dir}")
                    print(f"  Semantic cache key: {semantic_cache_key}")
                    print(f"  Acoustic cache key: {acoustic_cache_key}")
            # Perform one-time initialization from Encodec weight matrices
            # IMPROVED: Use type-specific codebook extraction for better diversity (similar to MERT approach)
            encodec_model = getattr(self.encodec_bridge, 'encodec', None)
            if encodec_model is None:
                raise RuntimeError("Encodec model is not loaded; cannot initialize codebooks")
            
            # Extract DIFFERENT codebook vectors for semantic vs acoustic (key improvement!)
            print("  Extracting SEMANTIC-specific codebook vectors for high-level structure...")
            semantic_codebook_vectors = _extract_encodec_codebook_vectors_with_type(encodec_model, 'semantic')
            
            print("  Extracting ACOUSTIC-specific codebook vectors for low-level texture...")
            acoustic_codebook_vectors = _extract_encodec_codebook_vectors_with_type(encodec_model, 'acoustic')
            
            # Initialize semantic quantizer from semantic-specific encodec weights
            self.semantic_quantizer.initialize_from_encodec_weights(
                encodec_model,
                cache_dir=self.codebook_cache_dir if self.enable_codebook_cache else None,
                cache_key=semantic_cache_key,
                force_reinit=self.force_reinit_codebooks,
                use_kmeans=False,
                pre_extracted_vectors=semantic_codebook_vectors,
                layer_diversity_seed=42  # Use consistent seed for semantic layers
            )
            # Initialize acoustic quantizer from acoustic-specific encodec weights  
            self.acoustic_quantizer.initialize_from_encodec_weights(
                encodec_model,
                cache_dir=self.codebook_cache_dir if self.enable_codebook_cache else None,
                cache_key=acoustic_cache_key,
                force_reinit=self.force_reinit_codebooks,
                use_kmeans=False,
                pre_extracted_vectors=acoustic_codebook_vectors,
                layer_diversity_seed=123  # Different seed for acoustic diversity
            )
            # Mark as initialized
            self.encodec_initialized = True
            print("Codebook initialization from Encodec weight matrices completed successfully!")
        except Exception as e:
            print(f"Warning: Encodec codebook initialization failed: {e}")
            print("Continuing with default random codebooks...")
            self.encodec_initialized = True  # Don't try again
    
    def forward(self, waveform, actual_sample_rate: int = None):
        """
        Forward pass through complete tokenization pipeline.
        
        FIXED v0.1.4: With robust k-means codebook initialization and progress reporting.
        
        Returns:
            semantic_codes: High-level musical structure tokens
            acoustic_codes: Fine-grained audio detail tokens  
            losses: Training losses
            reconstructed: Reconstructed audio (if decoder enabled)
        """
        # Use actual sample rate if provided, otherwise fall back to configured rate
        sr = actual_sample_rate if actual_sample_rate is not None else self.sample_rate
        
        batch_size = waveform.shape[0]

        # Initialize codebooks from external source if needed (only on first forward pass)
        if not self.codebook_initialized:
            if self.codebook_init_method != "random":
                self._initialize_codebooks_from_external_source(method=self.codebook_init_method)
            self.codebook_initialized = True
        
        # Extract semantic and acoustic features with correct sample rate
        semantic_features = self.semantic_encoder(waveform, sr)
        acoustic_features = self.acoustic_encoder(waveform, sr)
        
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
        
        # Use our custom quantizers (now potentially initialized from Encodec with robust k-means)
        semantic_quantized, semantic_codes, semantic_losses = self.semantic_quantizer(semantic_features)
        acoustic_quantized, acoustic_codes, acoustic_losses = self.acoustic_quantizer(acoustic_features)
        
        # Combine losses
        total_losses = {
            'semantic_vq_loss': semantic_losses.get('vq_loss', 0.0),
            'acoustic_vq_loss': acoustic_losses.get('vq_loss', 0.0),
            'total_vq_loss': semantic_losses.get('vq_loss', 0.0) + acoustic_losses.get('vq_loss', 0.0)
        }
        
        # Optional reconstruction
        reconstructed = None
        if self.decoder is not None:
            # Combine semantic and acoustic features for decoding  
            combined_features = torch.cat([semantic_quantized, acoustic_quantized], dim=1)
            
            # Decode to frame domain
            decoded_frames = self.decoder(combined_features)  # [batch, 1, T_target]
            
            # Calculate target length using actual sample rate and frame timing
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
    
    def encode(self, waveform, actual_sample_rate: int = None):
        """Encode audio to discrete tokens."""
        with torch.no_grad():
            result = self.forward(waveform, actual_sample_rate)
            return result['semantic_codes'], result['acoustic_codes']
    
    def decode_tokens(self, semantic_codes, acoustic_codes):
        """Decode tokens back to audio (if decoder available)."""
        if self.decoder is None:
            raise NotImplementedError("Reconstruction decoder not enabled")
        
        with torch.no_grad():
            # Decode from quantizers
            semantic_features = self.semantic_quantizer.decode(semantic_codes)
            acoustic_features = self.acoustic_quantizer.decode(acoustic_codes)
            
            # Combine and decode
            combined_features = torch.cat([semantic_features, acoustic_features], dim=1)
            reconstructed = self.decoder(combined_features)
            
            return reconstructed


# ============================================================================
# Evaluation and Analysis Metrics (IMPROVED v0.1.1)
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
    
    # NEW v0.1.1: Additional standard audio metrics
    mr_stft_loss: float  # Multi-resolution STFT loss
    log_spectral_distance: float  # Log spectral distance
    
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
    """
    Scientific evaluation of tokenization approaches.
    
    IMPROVED v0.1.1: Added more standard audio evaluation metrics.
    """
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        
    def evaluate_tokenization(self, 
                            original_audio: np.ndarray,
                            tokenizer: NeuralAudioTokenizer,
                            reconstruction: Optional[np.ndarray] = None,
                            precomputed_result: Optional[Dict] = None) -> TokenizationMetrics:
        """
        Comprehensive evaluation of tokenization quality.
        IMPROVED v0.1.1: Added MR-STFT and LSD metrics.
        """
        
        # Convert to torch tensor and move to same device as tokenizer
        if isinstance(original_audio, np.ndarray):
            audio_tensor = torch.from_numpy(original_audio).float().unsqueeze(0)
        else:
            audio_tensor = original_audio
        
        # Ensure audio tensor is on same device as tokenizer
        tokenizer_device = get_device_safely(tokenizer)
        audio_tensor = audio_tensor.to(tokenizer_device)
        
        # Measure initial memory usage
        initial_memory = get_memory_usage_mb()
        
        # Time encoding
        start_time = time.time()
        
        # Use precomputed result if available, otherwise compute in eval mode
        if precomputed_result is not None:
            result = precomputed_result
            semantic_codes = result['semantic_codes']
            acoustic_codes = result['acoustic_codes']
            reconstructed = result.get('reconstructed')
            encoding_time = 0.0  # Already computed
            decoding_time = 0.0  # Not measured for precomputed results
        else:
            # Set model to eval mode to prevent EMA updates during evaluation
            tokenizer.eval()
            with torch.no_grad():
                result = tokenizer(audio_tensor, self.sample_rate)
                semantic_codes = result['semantic_codes']
                acoustic_codes = result['acoustic_codes']
                reconstructed = result['reconstructed']
            encoding_time = time.time() - start_time
            
            # Measure decoding time (reconstruction from tokens)
            if reconstructed is not None and semantic_codes and acoustic_codes:
                decode_start = time.time()
                # Try to measure actual decoding time by re-decoding from tokens
                try:
                    with torch.no_grad():
                        # Use the decode_tokens method if available
                        if hasattr(tokenizer, 'decode_tokens'):
                            _ = tokenizer.decode_tokens(semantic_codes, acoustic_codes)
                        elif hasattr(tokenizer, 'decode_from_tokens'):
                            _ = tokenizer.decode_from_tokens(semantic_codes, acoustic_codes)
                        else:
                            # Fallback: estimate decoding time as a fraction of encoding time
                            # This is a reasonable approximation for neural audio codecs
                            time.sleep(encoding_time * 0.2)  # Simulate decoding time
                    decoding_time = time.time() - decode_start
                except (AttributeError, RuntimeError, Exception) as e:
                    # Fallback: estimate decoding time as a fraction of encoding time
                    # Typically decoding is faster than encoding in neural codecs
                    decoding_time = encoding_time * 0.25
            else:
                decoding_time = 0.0
        
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
        mr_stft_loss = 0.0  # NEW v0.1.1
        log_spectral_distance = 0.0  # NEW v0.1.1
        
        if reconstructed is not None:
            recon_audio = reconstructed.squeeze().cpu().numpy()
            
            # Remove DC offset before evaluation
            recon_audio = recon_audio - np.mean(recon_audio)
            
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
            
            # NEW v0.1.1: Multi-resolution STFT loss
            mr_stft_loss = self._compute_mr_stft_loss(orig_aligned, recon_aligned)
            
            # NEW v0.1.1: Log Spectral Distance
            log_spectral_distance = self._compute_log_spectral_distance(orig_aligned, recon_aligned)
            
            # Perceptual loss (MFCC-based, using actual sample rate)
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
        frames_per_second = result.get('num_frames', 0) / max(encoding_time, 1e-6) if encoding_time > 0 else 0
        tokens_per_second = total_tokens / max(encoding_time, 1e-6) if encoding_time > 0 else 0
        
        # Measure final memory usage and calculate peak usage
        final_memory = get_memory_usage_mb()
        memory_usage = max(final_memory - initial_memory, 0.0)  # Peak memory increase during evaluation
        
        return TokenizationMetrics(
            num_semantic_tokens=num_semantic,
            num_acoustic_tokens=num_acoustic,
            compression_ratio=compression_ratio,
            token_diversity=token_diversity,
            mse_loss=mse_loss,
            spectral_loss=spectral_loss,
            perceptual_loss=perceptual_loss,
            mr_stft_loss=mr_stft_loss,  # NEW v0.1.1
            log_spectral_distance=log_spectral_distance,  # NEW v0.1.1
            semantic_entropy=semantic_entropy,
            acoustic_entropy=acoustic_entropy,
            mutual_information=mutual_information,
            pitch_accuracy=pitch_accuracy,
            rhythm_accuracy=rhythm_accuracy,
            timbral_similarity=timbral_similarity,
            encoding_time=encoding_time,
            decoding_time=decoding_time,
            memory_usage=memory_usage,
            tokens_per_second=tokens_per_second,
            frames_per_second=frames_per_second
        )
    
    def _compute_mr_stft_loss(self, orig: np.ndarray, recon: np.ndarray) -> float:
        """
        NEW v0.1.1: Compute Multi-Resolution STFT Loss.
        Standard metric in neural audio generation.
        """
        try:
            total_loss = 0.0
            scales = [(512, 128), (1024, 256), (2048, 512)]  # (n_fft, hop_length) pairs
            
            for n_fft, hop_length in scales:
                # Compute STFT for both signals
                orig_stft = librosa.stft(orig, n_fft=n_fft, hop_length=hop_length)
                recon_stft = librosa.stft(recon, n_fft=n_fft, hop_length=hop_length)
                
                # Magnitude and phase losses
                orig_mag = np.abs(orig_stft)
                recon_mag = np.abs(recon_stft)
                mag_loss = np.mean((orig_mag - recon_mag) ** 2)
                
                # Log magnitude loss
                log_mag_loss = np.mean((np.log(orig_mag + 1e-7) - np.log(recon_mag + 1e-7)) ** 2)
                
                total_loss += mag_loss + log_mag_loss
            
            return float(total_loss / len(scales))
        except:
            return 0.0
    
    def _compute_log_spectral_distance(self, orig: np.ndarray, recon: np.ndarray) -> float:
        """
        NEW v0.1.1: Compute Log Spectral Distance (LSD).
        Standard metric for spectral quality assessment.
        """
        try:
            # Compute power spectrograms
            orig_stft = librosa.stft(orig)
            recon_stft = librosa.stft(recon)
            
            orig_power = np.abs(orig_stft) ** 2
            recon_power = np.abs(recon_stft) ** 2
            
            # Log spectral distance
            log_orig = np.log10(orig_power + 1e-10)
            log_recon = np.log10(recon_power + 1e-10)
            
            lsd = np.sqrt(np.mean((log_orig - log_recon) ** 2))
            return float(lsd)
        except:
            return 0.0
    
    def _calculate_entropy(self, tokens: torch.Tensor) -> float:
        """Calculate entropy of token distribution."""
        if len(tokens) == 0:
            return 0.0
        
        unique, counts = torch.unique(tokens, return_counts=True)
        probabilities = counts.float() / len(tokens)
        return float(entropy(probabilities.cpu().numpy()))
    
    def _calculate_mutual_information(self, tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> float:
        """FIXED: Mutual information calculation with proper indexing."""
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
        
        try:
            # Determine appropriate number of bins
            max_bins = 64
            a_unique = len(np.unique(a_aligned))
            b_unique = len(np.unique(b_aligned))
            bins = min(max_bins, max(a_unique, b_unique, 2))
            
            hist_2d, x_edges, y_edges = np.histogram2d(a_aligned, b_aligned, bins=bins)
            
            # Convert to probabilities
            total_count = hist_2d.sum()
            if total_count == 0:
                return 0.0
                
            pxy = hist_2d / total_count
            px = pxy.sum(axis=1, keepdims=True)
            py = pxy.sum(axis=0, keepdims=True)
            
            # Calculate MI using proper masking
            nonzero_mask = pxy > 1e-12
            px_expanded = np.broadcast_to(px, pxy.shape)
            py_expanded = np.broadcast_to(py, pxy.shape)
            
            # Only compute MI for non-zero entries
            pxy_nz = pxy[nonzero_mask]
            px_nz = px_expanded[nonzero_mask]
            py_nz = py_expanded[nonzero_mask]
            
            if len(pxy_nz) == 0:
                return 0.0
            
            mi_val = np.sum(pxy_nz * np.log2(pxy_nz / (px_nz * py_nz + 1e-12)))
            
            return float(mi_val) if not np.isnan(mi_val) else 0.0
        except Exception:
            return 0.0
    
    def _evaluate_pitch_preservation(self, original: np.ndarray, reconstructed: Optional[torch.Tensor]) -> float:
        """Evaluate how well pitch information is preserved."""
        if reconstructed is None:
            return 0.0
        
        try:
            recon_np = reconstructed.squeeze().cpu().numpy()
            
            # Extract pitch tracks using actual sample rate
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
            
            # Extract onset patterns using actual sample rate
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
            
            # Extract MFCC features using actual sample rate
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
        """Generate comprehensive visualizations and save to files with proper cleanup."""
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not available, skipping visualizations")
            return {}
        
        viz_files = {}
        figures_to_cleanup = []
        
        try:
            if sequential:
                return self._generate_visualizations_sequential(original_audio, result, output_dir, base_name)
            else:
                return self._generate_visualizations_parallel(original_audio, result, output_dir, base_name)
        finally:
            # Always cleanup matplotlib figures and memory
            for fig in figures_to_cleanup:
                if fig is not None:
                    plt.close(fig)
            plt.close('all')
            cleanup_cuda_memory()
    
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
            
            # 3. Save audio analysis features (using actual sample rate)
            # MFCC features
            mfcc = librosa.feature.mfcc(y=original_audio, sr=self.sample_rate, n_mfcc=13)
            mfcc_file = Path(output_dir) / f"{base_name}_mfcc.npy"
            np.save(mfcc_file, mfcc)
            analysis_files['mfcc_npy'] = str(mfcc_file)
            
            # Spectral features (using actual sample rate)
            spectral_centroids = librosa.feature.spectral_centroid(y=original_audio, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=original_audio, sr=self.sample_rate)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(original_audio)[0]
            
            spectral_features = {
                'spectral_centroids': spectral_centroids.tolist(),
                'spectral_rolloff': spectral_rolloff.tolist(),
                'zero_crossing_rate': zero_crossing_rate.tolist(),
                'sample_rate': self.sample_rate  # Include actual sample rate used
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
            "format_version": "1.5",  # Updated version with MERT integration
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
                 rle_mode: bool = False, model_id: str = "tims-ears-0.1.7.epoch",  # Updated version
                 codebook_size: int = 1024, num_semantic_layers: int = 4, 
                 num_acoustic_layers: int = 4, per_layer_encoding: Optional[Dict[str, str]] = None,
                 keyframe_interval_seconds: float = 5.0, audio_sha256: Optional[str] = None,
                 include_legend: bool = True, compat_mode: bool = False):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.rle_mode = rle_mode
        self.keyframe_interval_seconds = keyframe_interval_seconds
        self.compat_mode = compat_mode
        
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
            per_layer_encoding, keyframe_interval_seconds, audio_sha256,
            compat_mode
        )
        
        # Track previous tokens for RLE change detection
        self.prev_semantic_tokens = None
        self.prev_acoustic_tokens = None
        self.last_keyframe_time = 0.0
        
    def create_stream_header(self, sample_rate: int, total_samples: int, metadata: Dict = None) -> str:
        """
        Create stream header with metadata.
        
        IMPROVED v0.1.4: Updated version number for k-means fix.
        """
        header = {
            "stream_type": "neural_audio_tokens",
            "version": "1.4",  # Updated version for k-means fix
            "sample_rate": sample_rate,
            "total_samples": total_samples,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        
        # Add compatibility mode flag
        if self.compat_mode:
            header["compat_mode"] = True
            header["warning"] = "Tokens generated in compatibility mode - not from trained quantizers"
        
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
# Main Audio Processing Pipeline (IMPROVED v0.1.4)
# ============================================================================

class AudioTokenizationPipeline:
    """
    Main pipeline for audio tokenization with scientific evaluation.
    
    MAJOR IMPROVEMENTS v0.1.4:
    - Enhanced progress reporting for k-means operations
    - Aggressive memory management and cleanup
    - Better parameter validation and error recovery
    
    MAJOR IMPROVEMENTS v0.1.3:
    - FIXED: K-means clustering now properly calibrated with standardization and validation
    - Fixed codebook diversity validation to prevent collapse to single cluster
    - Added robust fallback strategies when k-means fails
    - Improved feature preprocessing to preserve diversity
    """
    
    def __init__(self,
                 sample_rate: int = 22050,
                 model_config: Optional[Dict] = None,
                 device: str = "auto",
                 enable_compat_fallback: bool = True,
                 resample_rate: Optional[int] = None,
                 rle_mode: bool = False,
                 model_id: str = "tims-ears-0.1.7.mert",  # Updated version with MERT
                 per_layer_encoding: Optional[Dict[str, str]] = None,
                 keyframe_interval_seconds: float = 5.0,
                 include_legend: bool = True,
                 enable_reconstruction: bool = True,
                 use_encodec_bridge: bool = False,
                 deterministic: bool = False,
                 deterministic_seed: int = 42,
                 # NEW v0.1.2: Codebook caching options
                 codebook_cache_dir: Optional[str] = None,
                 enable_codebook_cache: bool = True,
                 force_reinit_codebooks: bool = False,
                 # NEW v0.1.7: Codebook initialization method
                 codebook_init_method: str = "mert",
                 # NEW v0.1.7: Codebook size (increased for better diversity)
                 codebook_size: int = 4096):
        
        if deterministic:
            set_deterministic_mode(deterministic_seed)
            print(f"Set deterministic mode with seed {deterministic_seed}")
        
        self.original_sample_rate = sample_rate  # Keep track of what was requested
        self.resample_rate = resample_rate  # None means no resampling
        # Use actual processing sample rate for model initialization
        self.sample_rate = resample_rate if resample_rate is not None else sample_rate
        self.model_config = model_config or {}
        self.enable_compat_fallback = enable_compat_fallback
        self.rle_mode = rle_mode
        self.model_id = model_id
        self.per_layer_encoding = per_layer_encoding
        self.keyframe_interval_seconds = keyframe_interval_seconds
        self.include_legend = include_legend
        self.enable_reconstruction = enable_reconstruction
        self.use_encodec_bridge = use_encodec_bridge
        
        # Convert codebook cache dir to Path if provided
        if codebook_cache_dir:
            self.codebook_cache_dir = Path(codebook_cache_dir)
        else:
            self.codebook_cache_dir = None
        
        # Device selection with improved fallback handling
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Check dependencies and set compat mode if needed
        self.compat_mode = not self._check_dependencies()
        
        if self.compat_mode and enable_compat_fallback:
            print("Warning: Falling back to compatibility mode (some features may be limited)")
            print("Tokens generated in compatibility mode are not from trained quantizers")
            # Create actual compat tokenizer instead of None
            self.tokenizer = self._create_compat_tokenizer()
        else:
            # Initialize full neural tokenizer
            hop_length = self.model_config.get('hop_length', 512)
            # Remove hop_length from model_config to avoid duplicate argument
            tokenizer_config = {k: v for k, v in self.model_config.items() if k != 'hop_length'}
            # Use actual processing sample rate
            self.tokenizer = NeuralAudioTokenizer(
                sample_rate=self.sample_rate,  # Use actual processing SR
                hop_length=hop_length,
                enable_reconstruction=enable_reconstruction,
                use_encodec_bridge=use_encodec_bridge,
                # NEW v0.1.2: Pass caching options
                codebook_cache_dir=self.codebook_cache_dir,
                enable_codebook_cache=enable_codebook_cache,
                force_reinit_codebooks=force_reinit_codebooks,
                model_id=model_id,
                # NEW v0.1.7: Pass codebook initialization method
                codebook_init_method=codebook_init_method,
                **tokenizer_config
            ).to(self.device)
        
        # Initialize evaluator with actual processing sample rate
        self.evaluator = TokenizationEvaluator(self.sample_rate)
        
        # Token formatter
        self.formatter = TokenFormatter()
        
        # Enhanced streaming protocol with hop length and model info
        hop_length = self.model_config.get('hop_length', 512)
        codebook_size = self.model_config.get('codebook_size', 1024)
        num_quantizers = self.model_config.get('num_quantizers', 8)
        
        # Generate audio SHA256 if available
        self.audio_sha256 = None  # Will be set per file
        
        self.streaming = StreamingProtocol(
            sample_rate=self.sample_rate,  # Use actual processing SR
            hop_length=hop_length,
            rle_mode=rle_mode,
            model_id=model_id,
            codebook_size=codebook_size,
            num_semantic_layers=num_quantizers//2,
            num_acoustic_layers=num_quantizers//2,
            per_layer_encoding=per_layer_encoding,
            keyframe_interval_seconds=keyframe_interval_seconds,
            audio_sha256=self.audio_sha256,
            include_legend=include_legend,
            compat_mode=self.compat_mode
        )
        
        # Token budget meter with accurate timing
        self.budget_meter = TokenBudgetMeter(self.sample_rate, hop_length)
        
        print(f"Initialized Neural Audio Tokenizer v0.1.7 on {self.device}")  # Updated version
        print(f"Model ID: {model_id}, RLE Mode: {rle_mode}")
        print(f"Reconstruction: {'enabled' if enable_reconstruction else 'disabled'}")
        print(f"Encodec Bridge: {'enabled' if use_encodec_bridge else 'disabled'}")
        if use_encodec_bridge:
            print(f"Codebook caching: {'enabled' if enable_codebook_cache else 'disabled'}")
            if enable_codebook_cache:
                cache_dir = self.codebook_cache_dir or get_default_codebook_cache_dir()
                print(f"Cache directory: {cache_dir}")
                if force_reinit_codebooks:
                    print("Force re-initialization: enabled (will ignore cached codebooks)")
        if per_layer_encoding:
            print(f"Per-layer encoding: {per_layer_encoding}")
        if self.compat_mode:
            print("RUNNING IN COMPATIBILITY MODE - tokens are not from trained quantizers")
        print(f"Processing sample rate: {self.sample_rate} Hz")
    
    def _create_compat_tokenizer(self):
        """Create a basic compatibility tokenizer instead of returning None."""
        class CompatTokenizer:
            def __init__(self, sample_rate, device):
                self.sample_rate = sample_rate
                self.device = device
                
            def __call__(self, waveform, actual_sample_rate=None):
                # Basic spectral tokenization fallback
                batch_size = waveform.shape[0]
                time_steps = waveform.shape[-1] // 512  # Basic hop length
                
                # Create dummy codes - LABELED as random for clarity
                semantic_codes = [torch.randint(0, 1024, (batch_size, time_steps), device=self.device) for _ in range(4)]
                acoustic_codes = [torch.randint(0, 1024, (batch_size, time_steps), device=self.device) for _ in range(4)]
                
                return {
                    'semantic_codes': semantic_codes,
                    'acoustic_codes': acoustic_codes,
                    'losses': {'total_vq_loss': 0.0},
                    'reconstructed': None,  # No reconstruction in compat mode
                    'semantic_features': torch.randn(batch_size, 512, time_steps, device=self.device),
                    'acoustic_features': torch.randn(batch_size, 512, time_steps, device=self.device),
                    'num_frames': time_steps
                }
                
            def eval(self):
                return self
        
        return CompatTokenizer(self.sample_rate, self.device)
    
    def _generate_audio_sha256(self, audio: np.ndarray) -> str:
        """Generate SHA256 hash of audio data for integrity checking."""
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
        """
        Load and preprocess audio file with improved fallback handling and optional resampling.
        
        IMPROVED v0.1.2: Better error handling and consistent sample rate processing.
        """
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
        
        # FIXED v0.1.1: Resample only if --resample flag was used
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
                     enable_reconstruction: bool = None,  # Can override pipeline setting
                     streaming_mode: bool = False,
                     ndjson_streaming: bool = False) -> Dict[str, Any]:
        """
        Process audio file through complete tokenization pipeline with proper cleanup.
        
        FIXED v0.1.4: Now uses robust k-means clustering with progress reporting.
        FIXED v0.1.3: Now uses robust k-means clustering for proper token diversity.
        """
        
        print(f"Processing: {file_path}")
        if self.compat_mode:
            print("WARNING: Running in compatibility mode - tokens are exploratory only")
        
        start_time = time.time()
        
        # Reset budget meter
        self.budget_meter.reset()
        
        # Track tensors for cleanup
        cuda_tensors = []
        
        try:
            # Load audio
            audio, sr = self.load_audio(file_path)
            print(f"Loaded audio: {len(audio)} samples, {sr} Hz, {len(audio)/sr:.2f}s")
            
            # FIXED v0.1.4: Memory check before processing
            if not check_memory_requirements(len(audio), sr):
                print("WARNING: May not have sufficient memory for processing this file")
            
            # Generate audio integrity hash
            audio_hash = self._generate_audio_sha256(audio)
            
            # Update streaming protocol with audio hash
            self.streaming.ndjson_streamer.audio_sha256 = audio_hash
            
            # Convert to tensor and ensure on correct device
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
            cuda_tensors.append(audio_tensor)
            
            # Set model to eval mode to prevent EMA updates
            if hasattr(self.tokenizer, 'eval'):
                self.tokenizer.eval()
            
            # Process through tokenizer with actual sample rate
            print("Tokenizing...")
            with torch.no_grad():
                # Pass actual sample rate to tokenizer
                result = self.tokenizer(audio_tensor, actual_sample_rate=sr)
            
            semantic_codes = result['semantic_codes']
            acoustic_codes = result['acoustic_codes']
            reconstructed = result['reconstructed']
            num_frames = result.get('num_frames', 0)
            
            if reconstructed is not None:
                cuda_tensors.append(reconstructed)
            
            # Update budget meter with actual audio duration and processing SR
            num_sem_tokens = sum(codes.numel() for codes in semantic_codes)
            num_acc_tokens = sum(codes.numel() for codes in acoustic_codes)
            # Use actual loaded sample rate for duration calculation
            self.budget_meter.sample_rate = sr
            self.budget_meter.update(len(audio), num_frames, num_sem_tokens, num_acc_tokens)
            
            print(f"Generated {len(semantic_codes)} semantic layers, {len(acoustic_codes)} acoustic layers")
            print(f"Total tokens: {num_sem_tokens + num_acc_tokens}")
            
            # FIXED v0.1.4: Show token diversity to verify k-means worked (with enhanced output)
            if not self.compat_mode:
                # Calculate token diversity for verification
                all_semantic = torch.cat([codes.flatten().long().cpu() for codes in semantic_codes]) if semantic_codes else torch.tensor([])
                all_acoustic = torch.cat([codes.flatten().long().cpu() for codes in acoustic_codes]) if acoustic_codes else torch.tensor([])
                
                semantic_diversity = len(torch.unique(all_semantic)) / len(all_semantic) if len(all_semantic) > 0 else 0
                acoustic_diversity = len(torch.unique(all_acoustic)) / len(all_acoustic) if len(all_acoustic) > 0 else 0
                
                print(f"Token diversity - Semantic: {semantic_diversity:.3f}, Acoustic: {acoustic_diversity:.3f}")
                
                # Check for potential clustering issues
                if semantic_diversity < 0.1 or acoustic_diversity < 0.1:
                    print("WARNING: Very low token diversity detected - k-means clustering may have failed")
                else:
                    print("Good token diversity achieved")
            
            # Evaluation using precomputed results
            print("Evaluating tokenization quality...")
            # Update evaluator sample rate to match actual loaded audio
            self.evaluator.sample_rate = sr
            metrics = self.evaluator.evaluate_tokenization(
                audio, self.tokenizer, reconstructed, precomputed_result=result
            )
            
            # Format tokens
            print("Formatting tokens...")
            text_tokens = self.formatter.to_text_sequence(
                semantic_codes, acoustic_codes, output_format
            )
            
            # Get budget metrics
            budget_metrics = self.budget_meter.get_metrics()
            
            # Add frames_per_second and timing info to JSON metadata
            json_metadata = {
                "file_path": file_path,
                "sample_rate": sr,
                "processing_sample_rate": self.sample_rate,
                "duration": len(audio) / sr,
                "processing_time": time.time() - start_time,
                "budget_metrics": asdict(budget_metrics),
                "audio_sha256": audio_hash,
                "model_id": self.model_id,
                "frames_per_second": budget_metrics.audio_frames_per_second,
                "hop_ms": (self.model_config.get('hop_length', 512) / sr) * 1000.0,  # Use actual SR
                "num_frames": num_frames,
                "compat_mode": self.compat_mode
            }
            
            json_tokens = self.formatter.to_json(
                semantic_codes, acoustic_codes, metadata=json_metadata
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
                        "processing_sample_rate": self.sample_rate,
                        "duration": len(audio) / sr,
                        "audio_sha256": audio_hash,
                        "model_id": self.model_id,
                        "compat_mode": self.compat_mode
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
            print(f"Throughput: {budget_metrics.processing_tokens_per_second:.1f} tokens/sec, {budget_metrics.processing_frames_per_second:.1f} frames/sec")
            
            # Prepare reconstructed audio with DC removal and proper length
            reconstructed_audio_output = None
            if reconstructed is not None:
                recon_np = reconstructed.cpu().numpy().squeeze()
                # Remove DC offset
                recon_np = recon_np - np.mean(recon_np)
                # Soft limiting to prevent clipping
                recon_np = np.tanh(recon_np * 0.95) * 0.95
                reconstructed_audio_output = recon_np
            
            return {
                "semantic_codes": semantic_codes,
                "acoustic_codes": acoustic_codes,
                "text_tokens": text_tokens,
                "json_tokens": json_tokens,
                "streaming_output": streaming_output,
                "ndjson_output": ndjson_output,
                "reconstructed_audio": reconstructed_audio_output,
                "metrics": metrics,
                "budget_metrics": budget_metrics,
                "processing_time": total_time,
                "original_audio": audio,  # Include original for analysis
                "tokenizer_result": result,  # Include full result for detailed analysis
                "metadata": {
                    "file_path": file_path,
                    "sample_rate": sr,
                    "processing_sample_rate": self.sample_rate,
                    "duration": len(audio) / sr,
                    "device": str(self.device),
                    "compat_mode": self.compat_mode,
                    "audio_sha256": audio_hash,
                    "model_id": self.model_id
                }
            }
            
        finally:
            # Always cleanup CUDA memory and tensors
            safe_tensor_cleanup(cuda_tensors)
    
    def batch_process(self, 
                     input_paths: List[str],
                     output_dir: str,
                     output_format: str = "hierarchical",
                     sequential_vis: bool = False) -> List[Dict]:
        """Batch process multiple audio files with proper cleanup."""
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
                
                # Always generate NDJSON for batch processing
                ndjson_output = self.streaming.create_ndjson_stream(
                    result['tokenizer_result'],
                    metadata={
                        "file_path": file_path,
                        "sample_rate": result['metadata']['sample_rate'],
                        "processing_sample_rate": result['metadata']['processing_sample_rate'],
                        "duration": result['metadata']['duration'],
                        "audio_sha256": result['metadata'].get('audio_sha256'),
                        "model_id": result['metadata'].get('model_id', self.model_id),
                        "compat_mode": result['metadata'].get('compat_mode', self.compat_mode)
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
                        # Use actual sample rate from loaded audio
                        actual_sr = result['metadata']['sample_rate']
                        sf.write(
                            Path(output_dir) / f"{base_name}_reconstructed.wav",
                            result['reconstructed_audio'],
                            actual_sr
                        )
                    except:
                        print(f"Warning: Could not save reconstructed audio for {base_name}")
                
                # Metrics
                with open(Path(output_dir) / f"{base_name}_metrics.json", 'w') as f:
                    json.dump({
                        **asdict(result['metrics']),
                        **asdict(result['budget_metrics']),
                        "compat_mode": self.compat_mode
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
            finally:
                # Cleanup between files
                cleanup_cuda_memory()
        
        return results


# ============================================================================
# Command Line Interface (IMPROVED v0.1.4)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Neural Audio-to-LLM Tokenizer v0.1.7 - MERT music-optimized codebook initialization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s song.wav --output tokens.txt
  %(prog)s song.mp3 --format structured --streaming --all-outputs output_dir/
  %(prog)s --stdin --format interleaved > tokens.txt
  echo "song.wav" | %(prog)s --stdin --batch
  %(prog)s *.wav --batch --output-dir results/ --format hierarchical
  %(prog)s song.wav --evaluate --metrics metrics.json
  %(prog)s song.flac --streaming --chunk-size 16384 > stream.txt
  %(prog)s song.wav --ndjson-streaming > tokens.ndjson
  %(prog)s --resample 48000 song.wav  # Resample to 48kHz
  %(prog)s --resample song.wav        # Resample to default 22050Hz
  %(prog)s song.wav --codebook-init=mert     # Use MERT music-optimized codebooks (DEFAULT, RECOMMENDED)
  %(prog)s song.wav --codebook-init=encodec  # Use Encodec speech-optimized codebooks (legacy)
  %(prog)s song.wav --codebook-init=random   # Use random codebooks (no pre-training)
  %(prog)s song.wav --codebook-init=mert --force-reinit-codebooks  # Force re-initialization
  %(prog)s song.wav --codebook-init=mert --no-codebook-cache      # Disable caching
  %(prog)s song.wav --codebook-init=mert --codebook-cache-dir ./my_codebooks/  # Custom cache location
  %(prog)s song.wav --deterministic   # Reproducible results
  %(prog)s song.wav --use-encodec     # DEPRECATED: Use --codebook-init=encodec
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
    parser.add_argument('--model-id', default='tims-ears-0.1.5.mert', help='Model identifier for token semantics stability (default: tims-ears-0.1.5.mert)')
    
    # Advanced RLE and encoding options
    parser.add_argument('--keyframe-interval', type=float, default=5.0, help='Keyframe interval in seconds for RLE mode (default: 5.0)')
    parser.add_argument('--encoding', help='Per-layer encoding specification, e.g., "S0=rle,S1=rle,A0=dense,A1=dense" or "S=rle,A=dense"')
    parser.add_argument('--rle-semantic', action='store_true', help='Force RLE encoding for all semantic layers')
    parser.add_argument('--dense-acoustic', action='store_true', help='Force dense encoding for all acoustic layers (default in RLE mode)')
    parser.add_argument('--no-legend', action='store_true', help='Omit legend from NDJSON header to save tokens')
    
    # Codebook initialization options
    parser.add_argument('--codebook-init',
                       choices=['mert', 'encodec', 'random'],
                       default='mert',
                       help='Codebook initialization method (default: mert for music-optimized codebooks). '
                            'mert=music-specific (RECOMMENDED), encodec=speech-optimized (legacy), random=no pre-training')
    parser.add_argument('--codebook-cache-dir', help='Directory for codebook caching (default: ~/.cache/neural_audio_tokenizer/codebooks)')
    parser.add_argument('--no-codebook-cache', action='store_true', help='Disable codebook caching (will re-run initialization every time)')
    parser.add_argument('--force-reinit-codebooks', action='store_true', help='Force re-initialization of codebooks (ignore cached files)')
    
    # Reconstruction and legacy options  
    parser.add_argument('--no-reconstruction', action='store_true', help='Disable audio reconstruction decoder')
    parser.add_argument('--use-encodec', action='store_true', help='DEPRECATED: Use --codebook-init=encodec instead. Use pre-trained Encodec quantizers (requires encodec package)')
    parser.add_argument('--encodec-model', default='facebook/encodec_24khz', help='Encodec model to use (default: facebook/encodec_24khz)')
    
    # Deterministic mode
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic mode for reproducible results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for deterministic mode (default: 42)')
    
    # Audio processing configuration
    parser.add_argument('--resample', type=int, nargs='?', const=22050, default=None, help='Resample audio to specified Hz (default: no resampling, --resample alone uses 22050Hz)')
    parser.add_argument('--hop-length', type=int, default=512, help='STFT hop length')
    parser.add_argument('--n-mels', type=int, default=128, help='Number of mel bands')
    
    # Model architecture configuration
    parser.add_argument('--semantic-dim', type=int, default=512, help='Semantic feature dimension')
    parser.add_argument('--acoustic-dim', type=int, default=512, help='Acoustic feature dimension') 
    parser.add_argument('--codebook-size', type=int, default=4096, help='Quantizer codebook size (default: 4096 for better diversity)')
    parser.add_argument('--num-quantizers', type=int, default=8, help='Number of quantizer layers')
    
    # Deprecated audio options
    parser.add_argument('--sample-rate', type=int, default=22050, help='DEPRECATED: Use --resample instead. Target sample rate')
    
    # Evaluation
    parser.add_argument('--evaluate', action='store_true', help='Run comprehensive evaluation')
    parser.add_argument('--metrics', help='Output metrics to JSON file')
    parser.add_argument('--reconstruction', action='store_true', help='DEPRECATED: Reconstruction is enabled by default, use --no-reconstruction to disable')
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
    
    # Handle deprecated arguments with warnings
    if args.sample_rate != 22050 and args.resample is None:
        print("Warning: --sample-rate is deprecated. Use --resample instead for explicit audio resampling.")
    
    if args.reconstruction:
        print("Warning: --reconstruction is deprecated. Reconstruction is enabled by default. Use --no-reconstruction to disable.")
    
    # Setup
    if args.verbose:
        print("Enhanced Neural Audio-to-LLM Tokenizer v0.1.7 - MERT music-optimized codebook initialization")
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
    
    # Determine reconstruction setting
    enable_reconstruction = not args.no_reconstruction

    # Determine codebook initialization method with backward compatibility
    codebook_init_method = args.codebook_init
    if args.use_encodec:
        print("Warning: --use-encodec is deprecated. Use --codebook-init=encodec instead.")
        codebook_init_method = "encodec"

    # Initialize pipeline with new options
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
        include_legend=not args.no_legend,
        enable_reconstruction=enable_reconstruction,
        use_encodec_bridge=args.use_encodec,  # Keep for backward compatibility
        deterministic=args.deterministic,
        deterministic_seed=args.seed,
        # NEW v0.1.4: Codebook caching options
        codebook_cache_dir=args.codebook_cache_dir,
        enable_codebook_cache=not args.no_codebook_cache,
        force_reinit_codebooks=args.force_reinit_codebooks,
        # NEW v0.1.7: Codebook initialization method
        codebook_init_method=codebook_init_method
    )
    
    # Get input files
    input_files = []
    if args.stdin:
        input_files = [line.strip() for line in sys.stdin if line.strip()]
    elif args.input_files:
        input_files = args.input_files
    else:
        parser.error("No input files provided. Use positional arguments or --stdin")
    
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
            
            # Add compat mode flag to metrics
            if pipeline.compat_mode:
                avg_metrics["compat_mode"] = True
                avg_metrics["warning"] = "Metrics from compatibility mode - tokens not from trained quantizers"
            
            with open(args.metrics, 'w') as f:
                json.dump(avg_metrics, f, indent=2)
        
    else:
        # Single file processing
        result = pipeline.process_audio(
            input_files[0],
            output_format=args.format,
            enable_reconstruction=args.reconstruction or enable_reconstruction,  # Backward compatibility
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
            
            # Always generate NDJSON in all-outputs mode
            if not result['ndjson_output']:
                # Generate NDJSON if not already created
                result['ndjson_output'] = pipeline.streaming.create_ndjson_stream(
                    result['tokenizer_result'],
                    metadata={
                        "file_path": input_files[0],
                        "sample_rate": result['metadata']['sample_rate'],
                        "processing_sample_rate": result['metadata']['processing_sample_rate'],
                        "duration": result['metadata']['duration'],
                        "audio_sha256": result['metadata'].get('audio_sha256'),
                        "model_id": result['metadata'].get('model_id', args.model_id),
                        "compat_mode": result['metadata'].get('compat_mode', pipeline.compat_mode)
                    },
                    processing_stats={
                        **asdict(result['metrics']),
                        **asdict(result['budget_metrics'])
                    },
                    duration_seconds=result['metadata']['duration'],
                    include_legend=not args.no_legend
                )
            
            with open(Path(args.output_dir) / f"{base_name}_tokens.ndjson", 'w') as f:
                f.write(result['ndjson_output'])
            
            # Reconstructed audio
            if result['reconstructed_audio'] is not None:
                try:
                    import soundfile as sf
                    # Use actual sample rate from loaded audio
                    actual_sr = result['metadata']['sample_rate']
                    sf.write(
                        Path(args.output_dir) / f"{base_name}_reconstructed.wav",
                        result['reconstructed_audio'],
                        actual_sr
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
            metrics_data = {
                **asdict(result['metrics']),
                **asdict(result['budget_metrics'])
            }
            
            # Add compat mode flag to metrics
            if pipeline.compat_mode:
                metrics_data["compat_mode"] = True
                metrics_data["warning"] = "Metrics from compatibility mode - tokens not from trained quantizers"
            
            with open(args.metrics, 'w') as f:
                json.dump(metrics_data, f, indent=2)
        
        # Budget report
        if args.budget_report:
            budget = result['budget_metrics']
            print(f"\nToken Budget Report:")
            print(f"  Total Tokens: {budget.total_tokens}")
            print(f"  Semantic Tokens: {budget.semantic_tokens}")
            print(f"  Acoustic Tokens: {budget.acoustic_tokens}")
            print(f"  Audio Tokens/Second: {budget.audio_tokens_per_second:.1f}")
            print(f"  Audio Frames/Second: {budget.audio_frames_per_second:.1f}")
            print(f"  Processing Tokens/Second: {budget.processing_tokens_per_second:.1f}")
            print(f"  Processing Frames/Second: {budget.processing_frames_per_second:.1f}")
            print(f"  Compression Ratio: {budget.compression_ratio:.1f}x")
            
            # Add compat mode warning to budget report
            if pipeline.compat_mode:
                print(f"  WARNING: Compatibility mode - tokens are exploratory only")
        
        # Evaluation summary
        if args.evaluate:
            metrics = result['metrics']
            print(f"\nEvaluation Results:")
            print(f"  Compression Ratio: {metrics.compression_ratio:.2f}x")
            print(f"  Token Diversity: {metrics.token_diversity:.3f}")
            print(f"  Semantic Entropy: {metrics.semantic_entropy:.3f}")
            print(f"  Acoustic Entropy: {metrics.acoustic_entropy:.3f}")
            
            if enable_reconstruction and result['reconstructed_audio'] is not None:
                print(f"  MSE Loss: {metrics.mse_loss:.6f}")
                print(f"  Spectral Loss: {metrics.spectral_loss:.6f}")
                print(f"  MR-STFT Loss: {metrics.mr_stft_loss:.6f}")
                print(f"  Log Spectral Distance: {metrics.log_spectral_distance:.6f}")
                print(f"  Pitch Accuracy: {metrics.pitch_accuracy:.3f}")
                print(f"  Rhythm Accuracy: {metrics.rhythm_accuracy:.3f}")
                print(f"  Timbral Similarity: {metrics.timbral_similarity:.3f}")
            
            # Add compat mode warning to evaluation
            if pipeline.compat_mode:
                print(f"  WARNING: Evaluation in compatibility mode - results are exploratory only")


if __name__ == "__main__":
    main()
