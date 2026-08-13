import torch

from evlib.processing.reconstruction.e2vid_module.utils.loading_utils import get_device


def test_get_device_cuda(monkeypatch):
    # Simulate that CUDA is available.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)

    device = get_device(use_gpu=True)
    assert device.type == "cuda", f"Expected 'cuda', got {device}"


def test_get_device_mps(monkeypatch):
    # Simulate that CUDA is not available, but MPS is available and built.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)

    device = get_device(use_gpu=True)
    assert device.type == "mps", f"Expected 'mps', got {device}"


def test_get_device_cpu_no_gpu(monkeypatch):
    # Simulate that neither CUDA nor MPS is available.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)

    device = get_device(use_gpu=True)
    assert device.type == "cpu", f"Expected 'cpu', got {device}"


def test_get_device_cpu_when_not_requested(monkeypatch):
    # Simulate that GPUs are available but use_gpu is False.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)

    device = get_device(use_gpu=False)
    assert device.type == "cpu", f"Expected 'cpu' when use_gpu is False, got {device}"
