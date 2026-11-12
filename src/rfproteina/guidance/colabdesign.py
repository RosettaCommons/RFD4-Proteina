"""Experimental, first-order ColabDesign sequence-loss guidance.

The explicit value/gradient boundary lets JAX differentiate AlphaFold and PyTorch
differentiate the RFD4 decoder. No gradients pass through files or argmax sequences.
ColabDesign and its weights are optional and loaded only on the first active call.
"""

from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import torch
from atomworks.ml.conditions import C_SEQ
from torch.autograd.function import once_differentiable

from rfproteina.encoding import UNIFIED_ATOM37_ENCODING
from rfproteina.metrics.metric import Metric

# AlphaFold/ColabDesign order, explicitly mapped from RFD4's larger vocabulary.
AF_RESTYPES = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
)

SequenceObjective = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


class _ExternalSequenceLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, logits: torch.Tensor, target_aatype: torch.Tensor, objective: SequenceObjective
    ) -> torch.Tensor:
        # CPU copies are deliberate: portable across JAX/Torch devices and allocators.
        # This is an explicit VJP, not an accidental detach of the loss graph.
        value, gradient = objective(
            logits.detach().float().cpu().numpy().copy(),
            target_aatype.detach().cpu().numpy().copy(),
        )
        gradient = np.asarray(gradient)
        value = np.asarray(value)
        if value.shape != () or gradient.shape != tuple(logits.shape):
            raise ValueError("External objective must return a scalar and d(loss)/d(logits) with the input shape.")
        if not np.isfinite(value).all() or not np.isfinite(gradient).all():
            raise ValueError("Non-finite ColabDesign loss/gradient; refusing a silent zero-guidance step.")
        ctx.save_for_backward(torch.as_tensor(gradient.copy(), device=logits.device, dtype=logits.dtype))
        return logits.new_tensor(value.item())

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (gradient,) = ctx.saved_tensors
        return grad_output * gradient, None, None


class ColabDesignBinderLoss:
    """One deterministic AF2-multimer soft-sequence loss/gradient evaluation.

    Uses the public ``set_seq``/``run(backprop=True)`` API, without an optimizer
    step. The ``aux['grad']['seq']`` and prepared target sequence interfaces are
    version-sensitive; see the example README for the inspected upstream commit.
    """

    def __init__(
        self,
        target_pdb: str,
        data_dir: str,
        binder_length: int = 60,
        target_chain: str = "A",
        seed: int = 0,
        num_recycles: int = 0,
        weights: dict | None = None,
    ):
        if binder_length < 1 or num_recycles < 0:
            raise ValueError("binder_length must be positive and num_recycles nonnegative.")
        self.target_pdb = str(Path(target_pdb).resolve())
        self.data_dir = str(Path(data_dir).resolve())
        self.binder_length = binder_length
        self.target_chain = target_chain
        self.seed = seed
        self.num_recycles = num_recycles
        self.weights = dict(weights if weights is not None else {"plddt": 1.0, "i_pae": 1.0})
        if not self.weights or not all(np.isfinite(w) and w >= 0 for w in self.weights.values()):
            raise ValueError("Provide finite nonnegative ColabDesign loss weights.")
        if not any(self.weights.values()):
            raise ValueError("At least one loss weight must be positive.")
        self._model = None

    def _load(self) -> Any:
        if self._model is None:
            if not Path(self.target_pdb).is_file() or not Path(self.data_dir).is_dir():
                raise FileNotFoundError(
                    "Set target_pdb and data_dir to the prepared target PDB and AF2 weight directory."
                )
            from colabdesign import mk_afdesign_model

            model = mk_afdesign_model(
                protocol="binder",
                use_multimer=True,
                data_dir=self.data_dir,
                model_names=["model_1_multimer_v3"],
                num_seq=1,
                recycle_mode="last",
                clear_prev=True,
            )
            model.prep_inputs(
                pdb_filename=self.target_pdb, target_chain=self.target_chain, binder_len=self.binder_length
            )
            unknown = set(self.weights) - set(model.opt["weights"])
            if unknown:
                raise ValueError(f"Unknown ColabDesign loss weights: {sorted(unknown)}")
            # Remove protocol defaults before installing the requested loss, avoiding
            # hidden contact/PAE objectives. alpha=1 means the supplied values ARE logits.
            model.set_weights({key: 0.0 for key in model.opt["weights"]})
            model.set_weights(self.weights)
            model.set_opt(
                soft=1.0,
                hard=0.0,
                temp=1.0,
                alpha=1.0,
                dropout=False,
                num_models=1,
                sample_models=False,
                pssm_hard=False,
            )
            self._model = model
        return self._model

    def __call__(self, logits: np.ndarray, target_aatype: np.ndarray):
        if logits.shape != (1, self.binder_length, 20):
            raise ValueError(f"Expected one binder of {self.binder_length} residues, got {logits.shape}.")
        model = self._load()
        prepared = np.asarray(model._inputs["batch"]["aatype"])[: model._target_len]
        if not np.array_equal(prepared, target_aatype):
            raise ValueError("ColabDesign target sequence/order differs from the fixed RFD4 target.")
        model.set_seed(self.seed)
        model.set_seq(seq=logits, bias=np.zeros((self.binder_length, 20), dtype=np.float32))
        model.run(num_recycles=self.num_recycles, models=[0], num_models=1, sample_models=False, backprop=True)
        return np.asarray(model.aux["loss"]), np.asarray(model.aux["grad"]["seq"])


class ColabDesignSequenceMetric(Metric):
    """Loss on one fully designed protein chain against one sequence-fixed target.

    Selection comes from C_SEQ and chain IDs, not an assumed binder-first ordering.
    This intentionally rejects mixed polymers, atomization, unindexed motifs,
    partially fixed binders and multi-example batches rather than silently remapping.
    """

    differentiable = True

    def __init__(self, objective: SequenceObjective, **kwargs):
        super().__init__(**kwargs)
        self.objective = objective

    @cached_property
    def kwargs_to_compute_args(self) -> dict[str, tuple[str, str]]:
        return {"sequence_logits": ("decoded", "sequence_logits"), "batch": ("extra_info", "batch")}

    def compute(self, sequence_logits: torch.Tensor | None, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if sequence_logits is None or sequence_logits.ndim != 3 or sequence_logits.shape[0] != 1:
            raise ValueError("ColabDesign guidance requires AE sequence logits and batch_size=1.")
        if sequence_logits.shape[-1] != UNIFIED_ATOM37_ENCODING.n_tokens:
            raise ValueError("Expected the canonical RFD4 AE vocabulary.")
        valid = batch["padding_mask"][0].bool()
        if (batch["is_in_context_conditioning"][0] & valid).any():
            raise ValueError("Unindexed/context motifs are not supported by this example.")
        if not (batch["supports_local_latent"][0][valid].all() and (batch["token_type"][0][valid] == 0).all()):
            raise ValueError("This example supports only non-atomized standard protein residues.")
        sequence = batch[C_SEQ.full_name][0]
        binder = valid & (sequence == UNIFIED_ATOM37_ENCODING.token_to_idx["<M>"])
        target = valid & ~binder
        chains = batch["chain_iid"][0]
        if (
            not binder.any()
            or not target.any()
            or chains[binder].unique().numel() != 1
            or chains[target].unique().numel() != 1
            or chains[binder][0] == chains[target][0]
        ):
            raise ValueError("Require one fully sequence-free binder chain and one fully sequence-fixed target chain.")
        vocab = torch.tensor(
            [UNIFIED_ATOM37_ENCODING.token_to_idx[a] for a in AF_RESTYPES], device=sequence_logits.device
        )
        target_matches = sequence[target, None] == vocab[None, :]
        if not target_matches.any(-1).all():
            raise ValueError("Target contains a nonstandard amino acid unsupported by AF2.")
        target_aatype = target_matches.long().argmax(-1)
        logits = sequence_logits[:, binder].index_select(-1, vocab).float()
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite AE sequence logits.")
        return {"loss": _ExternalSequenceLoss.apply(logits, target_aatype, self.objective)}
