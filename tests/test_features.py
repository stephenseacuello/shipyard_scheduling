"""Tests for feature engineering."""

import numpy as np
from shipyard_scheduling.phm.feature_eng import (
    extract_features,
    extract_trending_features,
    extract_rate_of_change,
    extract_frequency_features,
    extract_all_features,
)


def test_extract_features():
    signal = [1.0, 2.0, 3.0, 4.0, 5.0]
    feats = extract_features(signal)
    assert feats["mean"] == 3.0
    assert feats["min"] == 1.0
    assert feats["max"] == 5.0
    assert feats["std"] > 0


def test_extract_features_empty():
    feats = extract_features([])
    assert feats["mean"] == 0.0


def test_trending_features():
    signal = list(range(30))  # linear trend
    feats = extract_trending_features(signal, window_sizes=[5, 10])
    assert feats["trend_slope_w5"] > 0
    assert feats["trend_slope_w10"] > 0


def test_rate_of_change():
    signal = [1.0, 3.0, 6.0, 10.0]
    feats = extract_rate_of_change(signal)
    assert feats["roc_mean"] == 3.0  # mean of [2, 3, 4]
    assert feats["acceleration_mean"] > 0


def test_frequency_features():
    # Sine wave should have a dominant frequency
    t = np.linspace(0, 1, 100)
    signal = list(np.sin(2 * np.pi * 10 * t))
    feats = extract_frequency_features(signal)
    assert feats["dominant_freq"] > 0
    assert feats["spectral_energy"] > 0


def test_extract_all_features():
    signal = list(range(30))
    feats = extract_all_features(signal)
    assert "mean" in feats
    assert "trend_slope_w5" in feats
    assert "roc_mean" in feats
    assert "dominant_freq" in feats
