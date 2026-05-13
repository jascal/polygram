"""`_select_panels` unit tests — greedy seeded coverage."""

from __future__ import annotations


import numpy as np

from polygram.compression.epoch import (
    _compute_cosine_graph,
    _select_panels,
)


def _synth_state(n_features: int = 16, d_model: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "W_enc": rng.standard_normal((d_model, n_features)).astype(np.float32),
        "b_enc": np.zeros(n_features, dtype=np.float32),
        "W_dec": rng.standard_normal((n_features, d_model)).astype(np.float32),
        "b_dec": np.zeros(d_model, dtype=np.float32),
    }


class TestPriorityOrder:
    def test_first_anchor_is_highest_priority(self):
        state = _synth_state(n_features=16)
        eligible = np.arange(16, dtype=np.int64)
        # Strictly decreasing priority by fid → first anchor must be 0.
        priority = np.linspace(10, 1, 16, dtype=np.float32)

        cosine_pairs = _compute_cosine_graph(
            state["W_dec"], eligible, threshold=-1.0
        )
        panels, _coverage = _select_panels(
            state_dict=state,
            eligible=eligible,
            priority=priority,
            cosine_pairs=cosine_pairs,
            zeroed=set(),
            n_visits_per_feature=4,
            n_panels_max=4,
            coverage_target=1.0,
            max_panel_size=8,
        )
        # First anchor is the highest-priority feature.
        assert panels[0].anchor == 0
        # Subsequent anchors come from features whose priority puts
        # them ahead of unused alternatives — not 0 again under
        # n_visits_per_feature=4 (the visit counter limits reuse).
        anchors = [p.anchor for p in panels]
        # No anchor exceeds n_visits_per_feature.
        from collections import Counter
        for fid, count in Counter(anchors).items():
            assert count <= 4


class TestSkipZeroed:
    def test_zeroed_features_never_appear(self):
        state = _synth_state(n_features=24)
        eligible = np.arange(24, dtype=np.int64)
        priority = np.ones(24, dtype=np.float32)
        cosine_pairs = _compute_cosine_graph(
            state["W_dec"], eligible, threshold=-1.0
        )
        zeroed = {0, 5, 10, 15, 20}
        panels, _coverage = _select_panels(
            state_dict=state,
            eligible=eligible,
            priority=priority,
            cosine_pairs=cosine_pairs,
            zeroed=zeroed,
            n_visits_per_feature=2,
            n_panels_max=10,
            coverage_target=1.0,
            max_panel_size=8,
        )
        for panel in panels:
            for fid in panel.feature_ids:
                assert int(fid) not in zeroed

    def test_zeroed_anchors_never_picked(self):
        """Even if zeroed features have high priority, they shouldn't
        be picked as anchors."""
        state = _synth_state(n_features=16)
        eligible = np.arange(16, dtype=np.int64)
        priority = np.zeros(16, dtype=np.float32)
        priority[3] = 100.0  # would otherwise be top
        priority[7] = 50.0
        cosine_pairs = _compute_cosine_graph(
            state["W_dec"], eligible, threshold=-1.0
        )
        zeroed = {3}
        panels, _coverage = _select_panels(
            state_dict=state,
            eligible=eligible,
            priority=priority,
            cosine_pairs=cosine_pairs,
            zeroed=zeroed,
            n_visits_per_feature=1,
            n_panels_max=2,
            coverage_target=1.0,
            max_panel_size=8,
        )
        anchors = [p.anchor for p in panels]
        assert 3 not in anchors


class TestVisitCap:
    def test_visit_cap_respected(self):
        state = _synth_state(n_features=16)
        eligible = np.arange(16, dtype=np.int64)
        priority = np.ones(16, dtype=np.float32)
        cosine_pairs = _compute_cosine_graph(
            state["W_dec"], eligible, threshold=-1.0
        )
        panels, _coverage = _select_panels(
            state_dict=state,
            eligible=eligible,
            priority=priority,
            cosine_pairs=cosine_pairs,
            zeroed=set(),
            n_visits_per_feature=2,
            n_panels_max=20,
            coverage_target=1.0,
            max_panel_size=8,
        )
        visits: dict[int, int] = {}
        for panel in panels:
            for fid in panel.feature_ids:
                visits[int(fid)] = visits.get(int(fid), 0) + 1
        for fid, count in visits.items():
            assert count <= 2, f"feature {fid} visited {count} times"


class TestCoverage:
    def test_coverage_achieves_target(self):
        state = _synth_state(n_features=16)
        eligible = np.arange(16, dtype=np.int64)
        priority = np.ones(16, dtype=np.float32)
        cosine_pairs = _compute_cosine_graph(
            state["W_dec"], eligible, threshold=-1.0
        )
        panels, coverage = _select_panels(
            state_dict=state,
            eligible=eligible,
            priority=priority,
            cosine_pairs=cosine_pairs,
            zeroed=set(),
            n_visits_per_feature=8,
            n_panels_max=100,
            coverage_target=0.95,
            max_panel_size=8,
        )
        # Target met or n_panels_max reached.
        assert coverage >= 0.95 or len(panels) == 100

    def test_empty_pair_graph_returns_full_coverage(self):
        state = _synth_state(n_features=16)
        eligible = np.arange(16, dtype=np.int64)
        priority = np.ones(16, dtype=np.float32)
        # Threshold beyond [0,1] guarantees empty pair set.
        cosine_pairs = _compute_cosine_graph(
            state["W_dec"], eligible, threshold=2.0
        )
        assert cosine_pairs == set()
        panels, coverage = _select_panels(
            state_dict=state,
            eligible=eligible,
            priority=priority,
            cosine_pairs=cosine_pairs,
            zeroed=set(),
            n_visits_per_feature=1,
            n_panels_max=2,
            coverage_target=0.95,
            max_panel_size=8,
        )
        # Target trivially met.
        assert coverage == 1.0


class TestDeterminism:
    def test_two_calls_produce_identical_panels(self):
        state = _synth_state(n_features=16)
        eligible = np.arange(16, dtype=np.int64)
        priority = np.ones(16, dtype=np.float32)
        # Tie-break on lowest fid; deterministic by construction.
        cosine_pairs = _compute_cosine_graph(
            state["W_dec"], eligible, threshold=-1.0
        )
        panels_a, _ = _select_panels(
            state_dict=state, eligible=eligible, priority=priority,
            cosine_pairs=cosine_pairs, zeroed=set(),
            n_visits_per_feature=2, n_panels_max=4, coverage_target=1.0,
            max_panel_size=8,
        )
        panels_b, _ = _select_panels(
            state_dict=state, eligible=eligible, priority=priority,
            cosine_pairs=cosine_pairs, zeroed=set(),
            n_visits_per_feature=2, n_panels_max=4, coverage_target=1.0,
            max_panel_size=8,
        )
        assert len(panels_a) == len(panels_b)
        for pa, pb in zip(panels_a, panels_b):
            assert pa.feature_ids == pb.feature_ids
            assert pa.anchor == pb.anchor


class TestPanelSize:
    def test_full_panels_have_eight_features(self):
        state = _synth_state(n_features=24)
        eligible = np.arange(24, dtype=np.int64)
        priority = np.ones(24, dtype=np.float32)
        cosine_pairs = _compute_cosine_graph(
            state["W_dec"], eligible, threshold=-1.0
        )
        panels, _ = _select_panels(
            state_dict=state, eligible=eligible, priority=priority,
            cosine_pairs=cosine_pairs, zeroed=set(),
            n_visits_per_feature=4, n_panels_max=2, coverage_target=1.0,
            max_panel_size=8,
        )
        for panel in panels:
            assert len(panel.feature_ids) == 8
