import pytest

from rfproteina.utils.device import prefer_eager_trunk


@pytest.mark.parametrize(
    ("device", "n", "env", "expected"),
    [
        ("mps", 512, None, False),
        ("mps", 640, None, True),
        ("mps", 640, "-1", False),
        ("mps", 256, "128", True),
        ("cuda", 1024, None, False),
        ("cpu", 1024, None, False),
    ],
)
def test_prefer_eager_trunk(monkeypatch, device, n, env, expected):
    if env is None:
        monkeypatch.delenv("RFP_MPS_EAGER_ABOVE", raising=False)
    else:
        monkeypatch.setenv("RFP_MPS_EAGER_ABOVE", env)
    assert prefer_eager_trunk(n, device) is expected


@pytest.mark.parametrize(
    ("requested", "env", "cuda", "mps", "expected"),
    [
        ("gpu", None, True, False, "cuda"),
        ("gpu", None, False, True, "mps"),
        ("auto", None, False, False, "cpu"),
        (None, None, False, True, "mps"),
        ("cpu", None, True, True, "cpu"),
        ("cuda", None, True, True, "cuda"),
        ("mps", None, False, True, "mps"),
        ("gpu", "cpu", False, False, "cpu"),  # RFP_DEVICE wins over the config
        ("cpu", "mps", False, True, "mps"),
    ],
)
def test_resolve_accelerator(monkeypatch, requested, env, cuda, mps, expected):
    from rfproteina.utils import device

    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(device, "_mps_available", lambda: mps)
    if env is None:
        monkeypatch.delenv("RFP_DEVICE", raising=False)
    else:
        monkeypatch.setenv("RFP_DEVICE", env)
    assert device.resolve_accelerator(requested) == expected


@pytest.mark.parametrize(
    ("requested", "cuda", "mps", "error"),
    [
        ("gpu", False, False, RuntimeError),  # no silent CPU fallback
        ("cuda", False, True, RuntimeError),
        ("mps", True, False, RuntimeError),
        ("tpu", True, True, ValueError),
    ],
)
def test_resolve_accelerator_rejects(monkeypatch, requested, cuda, mps, error):
    from rfproteina.utils import device

    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(device, "_mps_available", lambda: mps)
    monkeypatch.delenv("RFP_DEVICE", raising=False)
    with pytest.raises(error):
        device.resolve_accelerator(requested)


def test_autocast_dtype_follows_the_device_type():
    import torch

    from rfproteina.utils.device import autocast_dtype

    assert autocast_dtype("cpu") is None
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert autocast_dtype("cpu") is torch.bfloat16
        assert autocast_dtype("cuda") is None  # an autocast region on another device does not count
    assert autocast_dtype("cpu") is None
