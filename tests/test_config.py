"""Tests for ``polygram.config`` — tuning dataclasses, validation, and
dict round-trip.

Covers tasks.md §2.1–§2.4: frozen-ness / range failures / defaults,
round-trip equality, unknown-key warning, list→tuple coercion.
"""

from __future__ import annotations

import json
import warnings

import pytest

from polygram.config import (
    CancellationConfig,
    CompressionConfig,
    EpochCompressionConfig,
    RegrowConfig,
    SAEImportConfig,
    ValidationConfig,
)


_ALL_CONFIGS = [
    ValidationConfig,
    CancellationConfig,
    CompressionConfig,
    EpochCompressionConfig,
    SAEImportConfig,
]


# ---------------------------------------------------------------------------
# §2.1 — frozen-ness, range failures, defaults
# ---------------------------------------------------------------------------


class TestFrozen:
    @pytest.mark.parametrize("cls", _ALL_CONFIGS + [RegrowConfig])
    def test_dataclass_is_frozen(self, cls):
        assert cls.__dataclass_params__.frozen is True

    def test_assignment_raises(self):
        cfg = CompressionConfig()
        with pytest.raises(Exception):  # FrozenInstanceError subclasses Exception
            cfg.strategy = "zero"  # type: ignore[misc]

    def test_regrow_config_requires_keyword_only(self):
        # Positional construction is rejected because all fields are kw_only.
        with pytest.raises(TypeError):
            RegrowConfig("gpt2", 10)  # type: ignore[misc]


class TestDefaults:
    def test_validation_defaults(self):
        cfg = ValidationConfig()
        assert cfg.polygram_overlap_threshold == 0.7
        assert cfg.jaccard_threshold == 0.30
        assert cfg.min_firing_rate == 0.01
        assert cfg.min_both_fire == 5
        assert cfg.allow_layer_zero is False

    def test_cancellation_defaults(self):
        cfg = CancellationConfig()
        assert cfg.tolerance == 0.05
        assert cfg.preserve_tiers is True
        assert cfg.optimize == {"method": "grid", "max_steps": 50}
        assert cfg.grid_outer == (5, 5)
        assert cfg.min_amp_overlap == 0.0

    def test_compression_defaults(self):
        cfg = CompressionConfig()
        assert cfg.strategy == "merge"
        assert cfg.rep_selection == "scale_aware"
        assert cfg.merge_mode == "freq_weighted"
        assert cfg.confirmer is None
        assert cfg.target_n_features_kept is None
        assert cfg.score_field == "polygram_overlap"

    def test_epoch_compression_defaults_match_iterative_preset(self):
        cfg = EpochCompressionConfig()
        # Iterative-preset defaults (the new default after this change).
        assert cfg.coverage_target == 0.5
        assert cfg.cosine_threshold == 0.30
        assert cfg.n_visits_per_feature == 1
        assert cfg.max_iterations == 1
        assert cfg.quality_delta_multiplier == 2.0
        assert cfg.validation is None

    def test_sae_import_defaults_assign_gamma_true(self):
        cfg = SAEImportConfig()
        # Flipped from False to True per spec.
        assert cfg.assign_gamma is True
        assert cfg.gamma_range == (-0.25, 0.25)
        assert cfg.n_clusters == 2


class TestRangeValidation:
    def test_epoch_coverage_target_above_one_raises(self):
        with pytest.raises(ValueError, match=r"coverage_target.*\(0, 1\]"):
            EpochCompressionConfig(coverage_target=1.5)

    def test_epoch_coverage_target_zero_raises(self):
        with pytest.raises(ValueError, match=r"coverage_target.*\(0, 1\]"):
            EpochCompressionConfig(coverage_target=0.0)

    def test_epoch_cosine_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match=r"cosine_threshold.*\[-1, 1\]"):
            EpochCompressionConfig(cosine_threshold=1.5)

    def test_epoch_n_visits_zero_raises(self):
        with pytest.raises(ValueError, match=r"n_visits_per_feature"):
            EpochCompressionConfig(n_visits_per_feature=0)

    def test_epoch_max_iterations_zero_raises(self):
        with pytest.raises(ValueError, match=r"max_iterations"):
            EpochCompressionConfig(max_iterations=0)

    def test_validation_overlap_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match=r"polygram_overlap_threshold"):
            ValidationConfig(polygram_overlap_threshold=1.5)

    def test_validation_min_both_fire_negative_raises(self):
        with pytest.raises(ValueError, match=r"min_both_fire"):
            ValidationConfig(min_both_fire=-1)

    def test_cancellation_tolerance_above_one_raises(self):
        with pytest.raises(ValueError, match=r"tolerance"):
            CancellationConfig(tolerance=1.5)

    def test_cancellation_grid_outer_zero_raises(self):
        with pytest.raises(ValueError, match=r"grid_outer"):
            CancellationConfig(grid_outer=(0, 5))

    def test_cancellation_unknown_method_raises(self):
        with pytest.raises(ValueError, match=r"optimize\['method'\]"):
            CancellationConfig(optimize={"method": "bogus"})

    def test_compression_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match=r"strategy"):
            CompressionConfig(strategy="bogus")

    def test_compression_unknown_rep_selection_raises(self):
        with pytest.raises(ValueError, match=r"rep_selection"):
            CompressionConfig(rep_selection="bogus")

    def test_compression_target_n_features_kept_zero_raises(self):
        with pytest.raises(ValueError, match=r"target_n_features_kept"):
            CompressionConfig(target_n_features_kept=0)

    def test_compression_target_n_features_kept_negative_raises(self):
        with pytest.raises(ValueError, match=r"target_n_features_kept"):
            CompressionConfig(target_n_features_kept=-5)

    def test_compression_unknown_score_field_raises(self):
        with pytest.raises(ValueError, match=r"score_field"):
            CompressionConfig(score_field="bogus")

    def test_compression_kl_score_field_raises(self):
        # `kl_log_ratio_abs` is a real CandidatePair field but is
        # deliberately excluded from valid score_field values
        # (Decision 3 of add-pareto-target-compression).
        with pytest.raises(ValueError, match=r"score_field"):
            CompressionConfig(score_field="kl_log_ratio_abs")

    def test_regrow_layer_negative_raises(self):
        with pytest.raises(ValueError, match=r"layer"):
            RegrowConfig(model_name="gpt2", layer=-1)

    def test_regrow_empty_model_name_raises(self):
        with pytest.raises(ValueError, match=r"model_name"):
            RegrowConfig(model_name="", layer=10)

    def test_sae_import_inverted_gamma_range_raises(self):
        with pytest.raises(ValueError, match=r"gamma_range"):
            SAEImportConfig(gamma_range=(0.5, -0.5))


class TestRegrowConfigRequired:
    def test_missing_layer_raises_typeerror(self):
        with pytest.raises(TypeError, match=r"layer"):
            RegrowConfig(model_name="gpt2-medium")  # type: ignore[call-arg]

    def test_missing_model_name_raises_typeerror(self):
        with pytest.raises(TypeError, match=r"model_name"):
            RegrowConfig(layer=10)  # type: ignore[call-arg]

    def test_supplied_required_fields_succeeds(self):
        cfg = RegrowConfig(model_name="pythia-160m", layer=4)
        assert cfg.model_name == "pythia-160m"
        assert cfg.layer == 4
        assert cfg.strategy == "residual_kmeans"  # default


# ---------------------------------------------------------------------------
# §2.2 — round-trip equality
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_validation_round_trip(self):
        cfg = ValidationConfig(polygram_overlap_threshold=0.8, min_both_fire=10)
        assert ValidationConfig.from_dict(cfg.to_dict()) == cfg

    def test_cancellation_round_trip(self):
        cfg = CancellationConfig(
            tolerance=0.01,
            preserve_tiers=False,
            optimize={"method": "scipy", "max_steps": 200},
            grid_outer=(7, 9),
            min_amp_overlap=0.1,
        )
        assert CancellationConfig.from_dict(cfg.to_dict()) == cfg

    def test_compression_round_trip(self):
        cfg = CompressionConfig(strategy="zero", rep_selection="n_fires", confirmer="quantum_interference")
        assert CompressionConfig.from_dict(cfg.to_dict()) == cfg

    def test_compression_round_trip_with_target_k_fields(self):
        cfg = CompressionConfig(
            target_n_features_kept=500, score_field="jaccard"
        )
        assert CompressionConfig.from_dict(cfg.to_dict()) == cfg

    def test_compression_from_dict_tolerates_missing_target_k_fields(self):
        cfg = CompressionConfig.from_dict({
            "strategy": "merge",
            "rep_selection": "scale_aware",
            "merge_mode": "freq_weighted",
        })
        assert cfg.target_n_features_kept is None
        assert cfg.score_field == "polygram_overlap"

    def test_epoch_compression_round_trip_with_nested_validation(self):
        cfg = EpochCompressionConfig(
            coverage_target=0.7,
            max_iterations=3,
            validation=ValidationConfig(polygram_overlap_threshold=0.85),
        )
        d = cfg.to_dict()
        # Nested ValidationConfig serialises as a dict, not a dataclass instance.
        assert isinstance(d["validation"], dict)
        # And round-trips back to the original.
        assert EpochCompressionConfig.from_dict(d) == cfg

    def test_regrow_round_trip(self):
        cfg = RegrowConfig(
            model_name="pythia-160m",
            layer=4,
            strategy="residual_kmeans",
            prompts=("foo", "bar"),
            seed=42,
        )
        d = cfg.to_dict()
        # Tuple field serialises as a list inside the dict.
        assert isinstance(d["prompts"], list)
        # And round-trips back to a tuple-typed instance.
        cfg2 = RegrowConfig.from_dict(d)
        assert cfg2 == cfg
        assert isinstance(cfg2.prompts, tuple)

    def test_regrow_top_k_round_trip(self):
        cfg = RegrowConfig(model_name="pythia-160m", layer=4, top_k=3)
        d = cfg.to_dict()
        assert d["top_k"] == 3
        cfg2 = RegrowConfig.from_dict(d)
        assert cfg2 == cfg
        assert cfg2.top_k == 3

    def test_regrow_top_k_default_none_round_trip(self):
        cfg = RegrowConfig(model_name="pythia-160m", layer=4)
        d = cfg.to_dict()
        assert d["top_k"] is None
        cfg2 = RegrowConfig.from_dict(d)
        assert cfg2.top_k is None

    def test_sae_import_round_trip(self):
        cfg = SAEImportConfig(assign_gamma=False, gamma_range=(-0.1, 0.5), n_clusters=5)
        d = cfg.to_dict()
        assert isinstance(d["gamma_range"], list)
        cfg2 = SAEImportConfig.from_dict(d)
        assert cfg2 == cfg
        assert isinstance(cfg2.gamma_range, tuple)

    def test_sae_import_learn_axis_assignment_default(self):
        cfg = SAEImportConfig()
        assert cfg.learn_axis_assignment is None

    def test_sae_import_learn_axis_assignment_round_trip(self):
        cfg = SAEImportConfig(learn_axis_assignment=True)
        d = cfg.to_dict()
        cfg2 = SAEImportConfig.from_dict(d)
        assert cfg2.learn_axis_assignment is True

    def test_to_dict_is_json_serialisable(self):
        cfg = EpochCompressionConfig(
            coverage_target=0.7,
            validation=ValidationConfig(polygram_overlap_threshold=0.85),
        )
        # Should not raise.
        json.dumps(cfg.to_dict())


# ---------------------------------------------------------------------------
# §2.3 — unknown-key forward-compat warning
# ---------------------------------------------------------------------------


class TestUnknownKeyWarning:
    def test_unknown_top_level_key_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = CompressionConfig.from_dict(
                {"strategy": "merge", "futurefield": 42}
            )
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "futurefield" in str(w[0].message)
        # The known key still applied; defaults survive for the rest.
        assert cfg.strategy == "merge"
        assert cfg.rep_selection == "scale_aware"

    def test_unknown_key_in_nested_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = EpochCompressionConfig.from_dict(
                {
                    "coverage_target": 0.7,
                    "validation": {
                        "polygram_overlap_threshold": 0.8,
                        "futurenestedfield": True,
                    },
                }
            )
        # Exactly one warning, naming the nested unknown key.
        assert any("futurenestedfield" in str(wi.message) for wi in w)
        assert cfg.coverage_target == 0.7
        assert cfg.validation is not None
        assert cfg.validation.polygram_overlap_threshold == 0.8

    def test_from_dict_rejects_non_mapping(self):
        with pytest.raises(TypeError, match=r"mapping"):
            CompressionConfig.from_dict([("strategy", "merge")])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# §2.4 — list → tuple coercion for tuple-typed fields
# ---------------------------------------------------------------------------


class TestTupleCoercion:
    def test_grid_outer_list_becomes_tuple(self):
        cfg = CancellationConfig.from_dict({"grid_outer": [3, 4]})
        assert cfg.grid_outer == (3, 4)
        assert isinstance(cfg.grid_outer, tuple)

    def test_gamma_range_list_becomes_tuple(self):
        cfg = SAEImportConfig.from_dict({"gamma_range": [-0.1, 0.2]})
        assert cfg.gamma_range == (-0.1, 0.2)
        assert isinstance(cfg.gamma_range, tuple)

    def test_regrow_prompts_list_becomes_tuple(self):
        cfg = RegrowConfig.from_dict(
            {"model_name": "gpt2", "layer": 10, "prompts": ["a", "b"]}
        )
        assert cfg.prompts == ("a", "b")
        assert isinstance(cfg.prompts, tuple)


# ---------------------------------------------------------------------------
# Top-level re-export
# ---------------------------------------------------------------------------


class TestTopLevelImport:
    def test_polygram_top_level_re_exports_configs(self):
        import polygram

        assert polygram.CompressionConfig is CompressionConfig
        assert polygram.EpochCompressionConfig is EpochCompressionConfig
        assert polygram.CancellationConfig is CancellationConfig
        assert polygram.ValidationConfig is ValidationConfig
        assert polygram.RegrowConfig is RegrowConfig
        assert polygram.SAEImportConfig is SAEImportConfig
