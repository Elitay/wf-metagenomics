"""Test the diversity module."""

import numpy as np
import pandas as pd
import pytest
from workflow_glue.diversity import (
    global_rarefaction,
    random_choice,
    rarefaction_curve,
)


def _abundance_table():
    """Create a samtools depth table with dummy data."""
    data = {
        'barcode01': [10, 10, 2, 9],
        'barcode02': [100, 0, 0, 0],
        'barcode03': [50, 30, 5, 15],
        'barcode04': [0, 0, 0, 1]
    }
    df = pd.DataFrame(data)
    return df


def test_001_global_rarefaction():
    """Test global rarefaction with multivariate hypergeometric sampling."""
    abundance_table = _abundance_table()
    expected = {
        'barcode02': [95, 0, 0, 0],
        'barcode03': [49, 27, 5, 14],
    }
    expected_df = pd.DataFrame(expected)
    actual = global_rarefaction(df=abundance_table, seed=4)
    pd.testing.assert_frame_equal(
        actual,
        expected_df,
        check_dtype=True,
        check_exact=False,
        rtol=0.001,
    )


def test_002_random_choice_properties():
    """Subsample preserves length, sum, and per-taxon upper bounds."""
    arr = np.array([10, 10, 2, 9], dtype=np.int64)
    rarefied = random_choice(arr, 20, seed=4)
    assert len(rarefied) == len(arr)
    assert rarefied.sum() == 20
    assert np.all(rarefied <= arr)
    assert np.all(rarefied >= 0)


def test_003_random_choice_edge_cases():
    """Handle zero, full, and empty totals without expanding counts."""
    arr = np.array([10, 10, 2, 9], dtype=np.int64)
    np.testing.assert_array_equal(random_choice(arr, 0, seed=4), np.zeros_like(arr))
    np.testing.assert_array_equal(random_choice(arr, 31, seed=4), arr)
    np.testing.assert_array_equal(
        random_choice(np.zeros(3, dtype=np.int64), 5, seed=4),
        np.zeros(3, dtype=np.int64),
    )


def test_004_random_choice_memory_scale():
    """Large totals must not require O(N_reads) intermediate arrays."""
    arr = np.array([5_000_000, 3_000_000, 2_000_000], dtype=np.int64)
    rarefied = random_choice(arr, 1000, seed=1)
    assert rarefied.sum() == 1000
    assert np.all(rarefied <= arr)


@pytest.mark.parametrize("n_repeats,n_points", [(5, 10), (3, 5)])
def test_005_rarefaction_curve_shape(n_repeats, n_points):
    """Rarefaction curve returns the requested number of depth points."""
    counts = pd.Series([10, 10, 2, 9])
    curve = rarefaction_curve(counts, n_repeats=n_repeats, n_points=n_points)
    assert len(curve) == n_points
    assert 0 in curve
    assert curve[0] == 0
    assert max(curve.values()) <= int((counts > 0).sum())
