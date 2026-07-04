"""Tests for AgentMetrics after Phase 0.5 — fabricated token/cost removal."""
import pytest

from app.services.agent_service import AgentMetrics


class TestAgentMetrics:
    """Verify that AgentMetrics no longer fabricates token usage or costs."""

    def test_init_metrics_has_no_token_usage_or_costs(self):
        metrics = AgentMetrics.init_metrics()
        assert "token_usage" not in metrics
        assert "costs" not in metrics
        assert "node_timings" in metrics
        assert "cache_hits" in metrics
        assert "retrieval_stats" in metrics
        assert "workflow_start_time" in metrics

    def test_finalize_metrics_does_not_add_token_usage_or_costs(self):
        metrics = AgentMetrics.init_metrics()
        result = AgentMetrics.finalize_metrics(metrics)
        assert "token_usage" not in result
        assert "costs" not in result
        assert result["total_time"] >= 0
        assert "workflow_start_time" not in result

    def test_track_node_time_works(self):
        metrics = AgentMetrics.init_metrics()
        AgentMetrics.track_node_time(metrics, "understand_query", 0.42)
        assert metrics["node_timings"]["understand_query"] == 0.42

    def test_no_calculate_cost_method(self):
        """The old calculate_cost helper must be gone to prevent accidental reuse."""
        assert not hasattr(AgentMetrics, "calculate_cost")

    def test_no_track_token_usage_method(self):
        """The old track_token_usage helper must be gone to prevent accidental reuse."""
        assert not hasattr(AgentMetrics, "track_token_usage")
