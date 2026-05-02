"""Bootstrap smoke test — package imports and key dependency is reachable."""

import polygram


def test_package_has_version():
    assert isinstance(polygram.__version__, str)
    assert polygram.__version__


def test_q_orca_dependency_importable():
    import q_orca

    assert hasattr(q_orca, "__version__")


def test_q_orca_has_safe_rz_matcher():
    """The whole point of pinning q-orca>=0.7.1 — confirm the post-PR-#51
    matcher is reachable."""
    from q_orca.compiler.concept_gram_mps import (
        MpsGramConfigurationError,
        compute_concept_gram_mps,
    )

    assert callable(compute_concept_gram_mps)
    assert issubclass(MpsGramConfigurationError, Exception)
