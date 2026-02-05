"""Feature engineering utilities for prognostics.

This module provides helper functions to extract statistical, trending,
rate-of-change, and frequency-domain features from raw sensor sequences.
"""

from __future__ import annotations

from typing import Sequence, Dict, List
import numpy as np


def extract_features(signal: Sequence[float]) -> Dict[str, float]:
    """Return basic statistical features of a 1D sensor signal."""
    arr = np.asarray(signal, dtype=float)
    return {
        "mean": float(arr.mean()) if arr.size > 0 else 0.0,
        "std": float(arr.std()) if arr.size > 0 else 0.0,
        "min": float(arr.min()) if arr.size > 0 else 0.0,
        "max": float(arr.max()) if arr.size > 0 else 0.0,
    }


def extract_trending_features(
    signal: Sequence[float], window_sizes: List[int] | None = None
) -> Dict[str, float]:
    """Extract trending features: rolling mean slopes over multiple windows."""
    arr = np.asarray(signal, dtype=float)
    if window_sizes is None:
        window_sizes = [5, 10, 20]
    features: Dict[str, float] = {}
    for w in window_sizes:
        if arr.size >= w:
            rolling = np.convolve(arr, np.ones(w) / w, mode="valid")
            # Slope via linear regression on rolling means
            x = np.arange(len(rolling))
            if len(rolling) > 1:
                slope = float(np.polyfit(x, rolling, 1)[0])
            else:
                slope = 0.0
            features[f"trend_slope_w{w}"] = slope
        else:
            features[f"trend_slope_w{w}"] = 0.0
    return features


def extract_rate_of_change(signal: Sequence[float]) -> Dict[str, float]:
    """Extract rate-of-change features: first differences and acceleration."""
    arr = np.asarray(signal, dtype=float)
    features: Dict[str, float] = {}
    if arr.size >= 2:
        diff1 = np.diff(arr)
        features["roc_mean"] = float(diff1.mean())
        features["roc_std"] = float(diff1.std())
        features["roc_max"] = float(np.abs(diff1).max())
    else:
        features["roc_mean"] = 0.0
        features["roc_std"] = 0.0
        features["roc_max"] = 0.0
    if arr.size >= 3:
        diff2 = np.diff(arr, n=2)
        features["acceleration_mean"] = float(diff2.mean())
        features["acceleration_std"] = float(diff2.std())
    else:
        features["acceleration_mean"] = 0.0
        features["acceleration_std"] = 0.0
    return features


def extract_frequency_features(signal: Sequence[float]) -> Dict[str, float]:
    """Extract frequency-domain features via FFT."""
    arr = np.asarray(signal, dtype=float)
    features: Dict[str, float] = {}
    if arr.size >= 4:
        fft_vals = np.fft.rfft(arr - arr.mean())
        magnitudes = np.abs(fft_vals)
        freqs = np.fft.rfftfreq(arr.size)
        # Dominant frequency (excluding DC)
        if len(magnitudes) > 1:
            dom_idx = np.argmax(magnitudes[1:]) + 1
            features["dominant_freq"] = float(freqs[dom_idx])
            features["dominant_magnitude"] = float(magnitudes[dom_idx])
        else:
            features["dominant_freq"] = 0.0
            features["dominant_magnitude"] = 0.0
        features["spectral_energy"] = float(np.sum(magnitudes**2))
    else:
        features["dominant_freq"] = 0.0
        features["dominant_magnitude"] = 0.0
        features["spectral_energy"] = 0.0
    return features


def extract_all_features(
    signal: Sequence[float], window_sizes: List[int] | None = None
) -> Dict[str, float]:
    """Combine all feature types into a single dictionary."""
    features: Dict[str, float] = {}
    features.update(extract_features(signal))
    features.update(extract_trending_features(signal, window_sizes))
    features.update(extract_rate_of_change(signal))
    features.update(extract_frequency_features(signal))
    return features
