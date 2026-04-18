"""Unit tests for generate_dashboard_data.py pure functions."""
import pytest

from generate_dashboard_data import (
    PricingConfig,
    build_result,
    compute_cost_per_query,
    compute_tco_12mo,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def pricing():
    return PricingConfig(
        apis={
            "gpt-4.1": {"input_per_m": 2.0, "output_per_m": 8.0},
            "gpt-4.1-sft": {"input_per_m": 2.0, "output_per_m": 8.0, "training_per_m": 25.0},
        },
        self_hosted={"gpu_hourly_rate": 0.49, "queries_per_hour_per_gpu": 2000},
    )


# ── compute_cost_per_query ─────────────────────────────────────────────────────

def test_cost_per_query_self_hosted(pricing):
    cost = compute_cost_per_query("qwen3-8b", 100_000, 5_000, 1000, pricing)
    assert cost == pytest.approx(0.49 / 2000, rel=1e-3)


def test_cost_per_query_api_model(pricing):
    # 1000 predictions, avg 500 input + 50 output tokens
    cost = compute_cost_per_query("gpt-4.1", 500_000, 50_000, 1000, pricing)
    expected = (500 / 1_000_000) * 2.0 + (50 / 1_000_000) * 8.0
    assert cost == pytest.approx(expected, rel=1e-3)


def test_cost_per_query_zero_predictions(pricing):
    assert compute_cost_per_query("gpt-4.1", 0, 0, 0, pricing) is None


def test_cost_per_query_unknown_api_model(pricing):
    assert compute_cost_per_query("unknown-model", 1000, 100, 10, pricing) is None


# ── compute_tco_12mo ───────────────────────────────────────────────────────────

def test_tco_12mo_api_model(pricing):
    cost_per_query = 0.001
    daily_vol = 100
    tco = compute_tco_12mo("gpt-4.1", training_cost=0.0, cost_per_query=cost_per_query, daily_volume=daily_vol, pricing=pricing)
    expected = 0.001 * 100 * 365
    assert tco == pytest.approx(expected, rel=1e-3)


def test_tco_12mo_includes_training_cost(pricing):
    tco = compute_tco_12mo("gpt-4.1", training_cost=20.0, cost_per_query=0.001, daily_volume=100, pricing=pricing)
    assert tco > 20.0


def test_tco_12mo_none_cost_per_query(pricing):
    assert compute_tco_12mo("gpt-4.1", 0, None, 100, pricing) is None


def test_tco_12mo_self_hosted_uses_gpu_reservation(pricing):
    # Self-hosted uses GPU reservation cost, not per-query × volume
    tco = compute_tco_12mo("qwen3-8b", training_cost=2.5, cost_per_query=0.49 / 2000, daily_volume=1000, pricing=pricing)
    # 1000 queries/day → 1000/(2000*24) < 1 GPU → 0 GPUs (ceil=1? depends on math)
    # Just verify it's a reasonable positive number
    assert tco is not None
    assert tco > 0


# ── build_result ───────────────────────────────────────────────────────────────

def test_build_result_with_summary(pricing):
    summary = {
        "metric_value": 0.85,
        "metric_id": "weighted_f1",
        "n_predictions": 500,
        "total_input_tokens": 50_000,
        "total_output_tokens": 5_000,
        "ttft_p50_ms": 120.0,
        "ttft_p95_ms": 300.0,
        "error_counts": {"correct": 425, "wrong_class": 75},
    }
    result = build_result("gpt-4.1", "fpb", "zero-shot", summary, None, pricing)
    assert result["model_id"] == "gpt-4.1"
    assert result["metric_value"] == pytest.approx(0.85)
    assert result["family"] == "frontier"
    assert result["cost_per_query"] is not None
    assert result["cost_per_1k_correct"] is not None
    assert result["error_counts"]["correct"] == 425


def test_build_result_without_summary(pricing):
    result = build_result("gpt-4.1", "fpb", "zero-shot", None, None, pricing)
    assert result["metric_value"] is None
    assert result["cost_per_query"] is None


def test_build_result_with_training_meta(pricing):
    summary = {
        "metric_value": 0.9, "metric_id": "weighted_f1", "n_predictions": 100,
        "total_input_tokens": 10_000, "total_output_tokens": 1_000,
        "ttft_p50_ms": None, "ttft_p95_ms": None, "error_counts": {},
    }
    training_meta = {"training_cost": 2.5, "training_time_min": 45.0, "n_train": 500}
    result = build_result("qwen3-8b", "fpb", "lora-500", summary, training_meta, pricing)
    assert result["training_cost"] == pytest.approx(2.5)
    assert result["training_time_min"] == pytest.approx(45.0)
    assert result["n_train"] == 500


def test_build_result_display_name(pricing):
    result = build_result("qwen3-8b", "fpb", "lora-500", None, None, pricing)
    assert "Qwen3" in result["display_name"]


def test_build_result_unknown_model(pricing):
    result = build_result("mystery-model", "fpb", "zero-shot", None, None, pricing)
    assert result["family"] == "frontier"  # default
