"""EMA behavior tests: validation/inference use EMA weights, and resume == uninterrupted."""

import lightning as L
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rfproteina.generate import _load_ema_state_dict
from rfproteina.utils.simple_ema import SimpleEMA


class _TinyModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))

    def training_step(self, batch, batch_idx):
        x, y = batch
        return nn.functional.mse_loss(self.net(x), y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)


def _loader(seed: int = 0, n: int = 80, bs: int = 8) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    data = TensorDataset(torch.randn(n, 8, generator=g), torch.randn(n, 8, generator=g))
    return DataLoader(data, batch_size=bs, shuffle=False)


def _clone(sd) -> dict:
    return {k: v.clone() for k, v in sd.items()}


def _diverged_ema_callback(model: _TinyModule) -> SimpleEMA:
    """SimpleEMA whose EMA weights genuinely differ from the model's current (training) weights."""
    cb = SimpleEMA(decay=0.5, update_every_n=1)
    cb._ensure_ema_initialized(model)
    cb.ema_model.update_parameters(model)  # EMA := model_v0

    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)  # training weights := model_v0 + 1

    cb.ema_model.update_parameters(model)  # EMA := 0.5*model_v0 + 0.5*(model_v0+1) = model_v0 + 0.5
    return cb


def test_validation_uses_ema_and_restores_training_weights():
    model = _TinyModule()
    cb = _diverged_ema_callback(model)
    training = _clone(model.state_dict())
    ema = _clone(cb.ema_model.module.state_dict())
    assert any(not torch.equal(training[k], ema[k]) for k in training), "EMA must differ from training"

    cb.on_validation_start(None, model)
    assert all(torch.equal(model.state_dict()[k], ema[k]) for k in ema), "validation must run on EMA weights"

    cb.on_validation_end(None, model)
    assert all(torch.equal(model.state_dict()[k], training[k]) for k in training), "training weights must return"


def test_validation_exception_restores_training_weights():
    """A crash between on_validation_start and on_validation_end must not leave training on EMA weights."""
    model = _TinyModule()
    cb = _diverged_ema_callback(model)
    training = _clone(model.state_dict())

    cb.on_validation_start(None, model)
    cb.on_exception(None, model, RuntimeError("boom"))  # on_validation_end never fires
    assert all(torch.equal(model.state_dict()[k], training[k]) for k in training)
    assert cb._ema_weights_active is False


def test_ema_checkpoint_context_exposes_ema_weights():
    """EMAModelCheckpoint writes the -ema.ckpt inside use_ema_weights, so inference loading it gets EMA."""
    model = _TinyModule()
    cb = _diverged_ema_callback(model)
    training = _clone(model.state_dict())
    ema = _clone(cb.ema_model.module.state_dict())

    with cb.use_ema_weights(model):
        assert all(torch.equal(model.state_dict()[k], ema[k]) for k in ema)
    assert all(torch.equal(model.state_dict()[k], training[k]) for k in training)


def test_inference_loads_ema_weights_from_training_checkpoint(tmp_path):
    """A training checkpoint's EMA weights are extracted bare (keys match the model, no n_averaged)."""
    ckpt = tmp_path / "last.ckpt"
    weight = torch.randn(2, 3)
    torch.save({"callbacks": {"SimpleEMA": {"ema_state_dict": {"net.0.weight": weight}, "n_averaged": 7}}}, ckpt)

    sd = _load_ema_state_dict(str(ckpt))
    assert sd.keys() == {"net.0.weight"}
    assert torch.equal(sd["net.0.weight"], weight)


def test_inference_raises_when_checkpoint_has_no_ema(tmp_path):
    """Inference requires EMA weights, so a checkpoint without SimpleEMA state fails loudly."""
    ckpt = tmp_path / "no_ema.ckpt"
    torch.save({"callbacks": {}}, ckpt)
    with pytest.raises(ValueError, match="no SimpleEMA state"):
        _load_ema_state_dict(str(ckpt))


def _run(tmp_path, max_epochs: int, ckpt_in: str | None = None, ckpt_out: str | None = None):
    L.seed_everything(42, workers=True)
    model = _TinyModule()
    cb = SimpleEMA(decay=0.9, update_every_n=1)
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        devices=1,
        callbacks=[cb],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, _loader(), ckpt_path=ckpt_in)
    if ckpt_out:
        trainer.save_checkpoint(ckpt_out)
    return model, cb, trainer.optimizers[0].state_dict()


def _snapshot(model, cb, opt_state) -> dict:
    sd = cb.state_dict()
    return {
        "weights": _clone(model.state_dict()),
        "ema": _clone(sd["ema_state_dict"]),
        "n_averaged": int(sd["n_averaged"]),
        "opt": opt_state,
    }


def _assert_identical(a, b, path=""):
    if torch.is_tensor(a):
        assert a.shape == b.shape and torch.equal(a, b), f"tensor mismatch at {path}"
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), f"keys differ at {path}: {a.keys()} vs {b.keys()}"
        for k in a:
            _assert_identical(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list | tuple):
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_identical(x, y, f"{path}[{i}]")
    else:
        assert a == b, f"mismatch at {path}: {a} != {b}"


def test_resume_is_bitwise_identical_to_uninterrupted(tmp_path):
    """Train 2 epochs uninterrupted vs. train 1 epoch -> checkpoint -> resume -> train 1 more; the
    model weights, EMA weights, n_averaged counter, and optimizer state must be bitwise-identical --
    i.e. resuming is exactly like never having saved. Checkpoint at the epoch boundary so the
    (non-resumable) dataloader restarts cleanly and both runs see the same batches."""
    uninterrupted = _snapshot(*_run(tmp_path, max_epochs=2))

    ckpt = str(tmp_path / "epoch1.ckpt")
    _run(tmp_path, max_epochs=1, ckpt_out=ckpt)
    resumed = _snapshot(*_run(tmp_path, max_epochs=2, ckpt_in=ckpt))

    _assert_identical(uninterrupted["weights"], resumed["weights"], "weights")
    _assert_identical(uninterrupted["ema"], resumed["ema"], "ema")
    _assert_identical(uninterrupted["opt"], resumed["opt"], "optimizer")
    assert uninterrupted["n_averaged"] == resumed["n_averaged"]
