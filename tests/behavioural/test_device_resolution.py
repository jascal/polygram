"""`_resolve_device` — auto / explicit / fallback paths."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest

from polygram.behavioural.runtime import _resolve_device


def _torch_stub(*, cuda: bool, mps: bool):
    """Minimal stand-in matching the attribute lookups _resolve_device
    performs on the real `torch` module."""
    cuda_ns = SimpleNamespace(is_available=lambda: cuda)
    mps_ns = SimpleNamespace(is_available=lambda: mps)
    backends = SimpleNamespace(mps=mps_ns)
    return SimpleNamespace(cuda=cuda_ns, backends=backends)


class TestExplicit:
    def test_cpu_always_resolves(self):
        assert (
            _resolve_device(_torch_stub(cuda=False, mps=False), "cpu")
            == "cpu"
        )

    def test_cuda_resolves_when_available(self):
        assert (
            _resolve_device(_torch_stub(cuda=True, mps=False), "cuda")
            == "cuda"
        )

    def test_cuda_raises_when_unavailable(self):
        with pytest.raises(ValueError, match="no CUDA device"):
            _resolve_device(_torch_stub(cuda=False, mps=False), "cuda")

    def test_mps_resolves_when_available(self):
        assert (
            _resolve_device(_torch_stub(cuda=False, mps=True), "mps")
            == "mps"
        )

    def test_mps_raises_when_unavailable(self):
        with pytest.raises(ValueError, match="MPS backend is not"):
            _resolve_device(_torch_stub(cuda=False, mps=False), "mps")

    def test_unknown_device_raises(self):
        with pytest.raises(ValueError, match="unsupported device"):
            _resolve_device(_torch_stub(cuda=True, mps=True), "tpu")


class TestAuto:
    def test_auto_prefers_cuda(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert (
                _resolve_device(_torch_stub(cuda=True, mps=True), "auto")
                == "cuda"
            )

    def test_auto_falls_back_to_mps(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert (
                _resolve_device(_torch_stub(cuda=False, mps=True), None)
                == "mps"
            )

    def test_auto_falls_back_to_cpu_with_warning(self):
        with pytest.warns(RuntimeWarning, match="running on CPU"):
            d = _resolve_device(_torch_stub(cuda=False, mps=False), None)
        assert d == "cpu"

    def test_explicit_cpu_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            d = _resolve_device(_torch_stub(cuda=False, mps=False), "cpu")
        assert d == "cpu"
