"""EMA must follow optimizer updates, including after checkpoint restoration."""

from types import SimpleNamespace

import lightning as L
import pytest
import torch

from rfproteina.utils.simple_ema import SimpleEMA


@pytest.mark.parametrize("accumulation", [1, 2, 4])
@pytest.mark.parametrize("interval", [1, 2])
@pytest.mark.parametrize("restored_step", [0, 4])
def test_ema_updates_once_per_optimizer_step(accumulation, interval, restored_step):
    model = L.LightningModule()
    model.weight = torch.nn.Parameter(torch.tensor([0.0]))
    trainer = SimpleNamespace(global_step=restored_step)
    callback = SimpleEMA(decay=0.5, update_every_n=interval)
    callback.on_fit_start(trainer, model)
    callback.on_train_start(trainer, model)
    expected = None
    count = 0
    for step in range(1, 5):
        for micro in range(accumulation):
            if micro == accumulation - 1:
                trainer.global_step += 1
                with torch.no_grad():
                    model.weight.fill_(step)
                if trainer.global_step % interval == 0:
                    expected = float(step) if expected is None else 0.5 * expected + 0.5 * step
                    count += 1
            callback.on_train_batch_end(trainer, model, {}, {}, (step - 1) * accumulation + micro)
            assert callback.ema_model.n_averaged.item() == count
            if expected is not None:
                torch.testing.assert_close(callback.ema_model.module.weight, torch.tensor([expected]))
