"""Tests for metrics computation."""

from shipyard_scheduling.utils.metrics import compute_kpis, compute_episode_summary


def test_compute_kpis_basic():
    metrics = {
        "blocks_completed": 10,
        "total_tardiness": 50.0,
        "breakdowns": 2,
        "planned_maintenance": 3,
        "n_spmts": 6,
        "n_cranes": 2,
        "spmt_busy_time": 300.0,
        "crane_busy_time": 100.0,
        "completion_deltas": [1.0, 2.0, 3.0, 4.0],
        "empty_travel_time": 20.0,
    }
    kpis = compute_kpis(metrics, total_time=1000.0)
    assert kpis["throughput"] == 10 / 1000.0
    assert kpis["average_tardiness"] == 50.0 / 10
    assert kpis["spmt_utilization"] == 300.0 / (6 * 1000.0)
    assert kpis["crane_utilization"] == 100.0 / (2 * 1000.0)
    assert kpis["oee"] >= 0
    assert kpis["schedule_variance"] > 0
    assert kpis["total_cost"] > 0


def test_compute_kpis_zero_time():
    kpis = compute_kpis({"blocks_completed": 0}, total_time=0.0)
    assert kpis["throughput"] == 0.0


def test_episode_summary():
    eps = [
        {"throughput": 1.0, "tardiness": 2.0},
        {"throughput": 3.0, "tardiness": 4.0},
    ]
    summary = compute_episode_summary(eps)
    assert summary["throughput"] == 2.0
    assert summary["tardiness"] == 3.0
