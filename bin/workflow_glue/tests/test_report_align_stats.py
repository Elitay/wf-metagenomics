"""Test the table generation code.

Test the alignment statistic helper functions
"""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import pandas as pd
import pytest
from workflow_glue.alignment_stats import (
    alignment_metrics,
    depth2heatmap,
)
from workflow_glue.report_utils.report_utils import (
    filter_and_cap_alignment_frames,
    prepare_alignment_frames,
    prepare_alignment_job,
    resolve_alignment_rank,
)


@pytest.fixture
def valid_inputs_dir(request):
    """Define data location fixture."""
    return Path(
        request.config.getoption(
            "--test_data")) / "workflow_glue/case03/alignment_stats"


def _depth_table():
    """Create a samtools depth table with dummy data."""
    data = {
        'ref': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'C'],
        'pos': [2, 3, 5, 7, 9, 1, 2, 3, 1],
        'depth': [2, 2, 1, 2, 3, 4, 4, 4, 1]
    }
    df = pd.DataFrame(data)
    return df


def _cov_table():
    """Create a simplified samtools coverage table with dummy data."""
    data = {
        'ref': ['B', 'A', 'C'],
        'endpos': [3, 10, 4],
    }
    df = pd.DataFrame(data)
    return df.set_index('ref')


@pytest.mark.parametrize(
    "expected", [
        {
            'ref': ['B', 'A', 'C'],
            'endpos': [3, 10, 4],
            'mean': [4.0, 1.0, 0.25],
            'sd': [0.0, 1.095, 0.433],
            'Coefficient of Variance': [0.0, 1.095, 1.732],
        }
    ]
)
def test_001_alignment_metrics(expected):
    """Test alignment_metrics function."""
    depth_df = _depth_table()
    cov_df = _cov_table()

    expected_df = pd.DataFrame(expected)
    actual = alignment_metrics(depth=depth_df, stats=cov_df)
    pd.testing.assert_frame_equal(
        actual,
        expected_df,
        check_dtype=True,
        check_categorical=True,
        check_exact=False,
        rtol=0.001,
        check_datetimelike_compat=True,
    )


@pytest.mark.parametrize(
    "windows, expected", [
        # windows = 1, the results are the same as the mean_depth from alignment metrics
        # no percentiles, just 1
        (1, {
            'ref': ['B', 'A'],
            0: [4.0, 1.0],
        }),
        (2, {
            'ref': ['B', 'A'],
            0: [8.0, 1.0],
            1: [4.0, 1.0],
        }),
    ]
)
def test_002_depth2heatmap(windows, expected):
    """Test depth2heatmap function."""
    depth_df = _depth_table()
    cov_df = _cov_table()
    expected_df = pd.DataFrame(expected).set_index('ref')
    expected_df.columns = [int(val) for val in expected_df.columns]
    actual = depth2heatmap(depth=depth_df, reference=cov_df, windows=windows)
    pd.testing.assert_frame_equal(
        actual,
        expected_df,
        check_dtype=True,
        check_categorical=True,
        check_exact=False,
        rtol=0.001,
        check_datetimelike_compat=True,
    )


def _align_and_heatmap_frames(n_refs=5):
    """Build dummy alignment and heatmap frames."""
    refs = [f"ref{i}" for i in range(1, n_refs + 1)]
    species = [f"Sp{i}" for i in range(1, n_refs + 1)]
    align_df = pd.DataFrame({
        "reference": refs,
        "ref length": [1000] * n_refs,
        "number of reads": list(range(n_refs, 0, -1)),
        "% coverage": [90.0] * n_refs,
        "mean depth": list(range(n_refs, 0, -1)),
        "mean baseq": [20.0] * n_refs,
        "mean MAPQ": [40.0] * n_refs,
        "sd depth": [1.0] * n_refs,
        "CV": [0.1] * n_refs,
        "% reads": [10.0] * n_refs,
        "species": species,
        "genus": ["Genus"] * n_refs,
    })
    heatmap = pd.DataFrame(
        {w: [float(r + 1) for r in range(n_refs)] for w in range(3)},
        index=refs,
    )
    return align_df, heatmap


def test_101_filter_and_cap_alignment_frames():
    """Cap heatmap to top-N refs by mean depth and filter taxa."""
    align_df, heatmap = _align_and_heatmap_frames(n_refs=5)
    filtered, capped = filter_and_cap_alignment_frames(
        align_df, heatmap, rank_col="species",
        allowed_taxa=["Sp1", "Sp2", "Sp3", "Sp5"],
        n_heatmap_refs=2,
    )
    assert set(filtered["species"]) == {"Sp1", "Sp2", "Sp3", "Sp5"}
    # Highest mean depth among allowed+heatmap: Sp1 (5), Sp2 (4)
    assert list(capped.index) == ["ref1", "ref2"]
    assert len(capped) == 2


def test_102_resolve_alignment_rank_falls_back_to_genus():
    """Fall back to genus when species column is absent."""
    align_df, _ = _align_and_heatmap_frames(n_refs=2)
    align_df = align_df.drop(columns=["species"])
    assert resolve_alignment_rank(align_df, "S") == "genus"
    assert resolve_alignment_rank(align_df, "G") == "genus"


def test_103_prepare_alignment_frames_from_tsv(tmp_path):
    """Read alignment TSVs and return capped frames."""
    sample = "barcode01"
    align_df, heatmap = _align_and_heatmap_frames(n_refs=4)
    # Match on-disk column names expected by prepare_alignment_frames
    raw_align = align_df.rename(columns={
        "reference": "ref",
        "ref length": "endpos",
        "mean depth": "meandepth",
        "mean baseq": "meanbaseq",
        "mean MAPQ": "meanmapq",
        "sd depth": "sd",
        "CV": "Coefficient of Variance",
        "% reads": "pcreads",
    })
    raw_align["startpos"] = 0
    raw_align.to_csv(tmp_path / f"{sample}-alignment-stats.tsv", sep="\t", index=False)
    heatmap.to_csv(tmp_path / f"{sample}-alignment-stats-heatmap.tsv", sep="\t")

    prepared = prepare_alignment_frames(
        str(tmp_path), sample, taxonomic_rank="S",
        allowed_taxa=["Sp1", "Sp2", "Sp3"], n_heatmap_refs=2,
    )
    assert prepared is not None
    out_align, out_heatmap, rank_col = prepared
    assert rank_col == "species"
    assert list(out_heatmap.index) == ["ref1", "ref2"]
    assert set(out_align["species"]).issubset({"Sp1", "Sp2", "Sp3"})


def test_104_prepare_alignment_job_pool(tmp_path):
    """Smoke-test process-pool prep path with spawn context."""
    samples = ["barcode01", "barcode02"]
    for i, sample in enumerate(samples):
        align_df, heatmap = _align_and_heatmap_frames(n_refs=3)
        align_df["species"] = [f"Sp{j}" for j in range(1, 4)]
        raw_align = align_df.rename(columns={
            "reference": "ref",
            "ref length": "endpos",
            "mean depth": "meandepth",
            "mean baseq": "meanbaseq",
            "mean MAPQ": "meanmapq",
            "sd depth": "sd",
            "CV": "Coefficient of Variance",
            "% reads": "pcreads",
        })
        raw_align["startpos"] = 0
        # Slightly different depths per sample
        raw_align["meandepth"] = [10 - i, 5, 1]
        raw_align.to_csv(
            tmp_path / f"{sample}-alignment-stats.tsv", sep="\t", index=False)
        heatmap.to_csv(
            tmp_path / f"{sample}-alignment-stats-heatmap.tsv", sep="\t")

    jobs = [
        (str(tmp_path), sample, "S", ["Sp1", "Sp2"], 2)
        for sample in samples
    ]
    with ProcessPoolExecutor(
        max_workers=2, mp_context=get_context("spawn")
    ) as pool:
        results = dict(pool.map(prepare_alignment_job, jobs))
    assert set(results) == set(samples)
    for prepared in results.values():
        assert prepared is not None
        _, out_heatmap, _ = prepared
        assert len(out_heatmap) <= 2
