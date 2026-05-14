import sys
from types import SimpleNamespace

from text_distillation.utils import get_device


def _torch_with_devices(*, cuda: bool = False, mps_built: bool = False, mps_available: bool = False):
    mps = SimpleNamespace(
        is_built=lambda: mps_built,
        is_available=lambda: mps_available,
    )
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=mps),
    )


def test_get_device_prefers_cuda_over_mps(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _torch_with_devices(cuda=True, mps_built=True, mps_available=True),
    )

    assert get_device() == "cuda"


def test_get_device_uses_mps_before_cpu(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _torch_with_devices(cuda=False, mps_built=True, mps_available=True),
    )

    assert get_device() == "mps"


def test_get_device_can_disable_mps(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _torch_with_devices(cuda=False, mps_built=True, mps_available=True),
    )

    assert get_device(prefer_mps=False) == "cpu"


def test_get_device_falls_back_to_cpu_when_mps_unavailable(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _torch_with_devices(cuda=False, mps_built=True, mps_available=False),
    )

    assert get_device() == "cpu"
