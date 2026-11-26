"""What the full-BioIR backend refuses to bridge from native beam search.

These cover the rejections that read the native strategy alone, so they run
wherever ``rfproteina`` imports, with or without ``bionemo_ir`` installed.
"""

import importlib.util
from types import SimpleNamespace

import pytest
import torch

from rfproteina.nn.bioir_full import search_from_native
from rfproteina.search.beam_search import BeamSearchStrategy

BIOIR_INSTALLED = importlib.util.find_spec("bionemo_ir") is not None


def _seam_available() -> bool:
    """Whether the installed BioIR carries the resume seam, not merely whether it is installed."""
    if not BIOIR_INSTALLED:
        return False
    import inspect

    from bionemo_ir.models.rfproteina import RFProteina

    from rfproteina.nn.bioir_full import _SEAM_PARAMETERS

    return set(inspect.signature(RFProteina.search).parameters) >= _SEAM_PARAMETERS


SEAM_AVAILABLE = _seam_available()
needs_seam = pytest.mark.skipif(not SEAM_AVAILABLE, reason="needs a bionemo-ir carrying the resume seam")


def _native(strategy, nsteps: int = 60):
    return SimpleNamespace(search_strategy=strategy, inf_cfg=SimpleNamespace(args=SimpleNamespace(nsteps=nsteps)))


def _strategy(**kwargs) -> BeamSearchStrategy:
    return BeamSearchStrategy({40: {}, 60: {}}, n_branch=2, beam_width=2, **kwargs)


def test_no_search_strategy_is_not_a_search() -> None:
    assert search_from_native(_native(None)) is None


@needs_seam
def test_export_candidates_stops_at_the_first_interior_checkpoint() -> None:
    """The caller replaces the prune at step 40, so BioIR has no reason to run past it."""
    config = search_from_native(_native(_strategy(export_candidates=True)))
    assert config.step_checkpoints == [0, 40]


@needs_seam
def test_a_native_export_is_refused_on_this_backend(tmp_path) -> None:
    """Both backends write *_candidates.pt; only one of them carries x_1."""
    path = tmp_path / "candidates.pt"
    torch.save({"candidates": {}, "step": 40, "example_ids": ["a"], "nsamples": 1, "backend": "native"}, path)
    strategy = _strategy(resume_from=str(path), resume_seed=7, resume_selection=[0, 1])
    with pytest.raises(NotImplementedError, match="carries no x_1 predictions"):
        search_from_native(_native(strategy))


@pytest.mark.skipif(not BIOIR_INSTALLED, reason="needs bionemo_ir installed to inspect its search()")
def test_the_seam_release_is_named_when_bioir_is_too_old(monkeypatch) -> None:
    """A user on 0.2.0 is told which release to move to, not that the feature does not exist."""
    import rfproteina.nn.bioir_full as bridge

    monkeypatch.setattr(bridge, "_SEAM_PARAMETERS", frozenset({"a_parameter_search_will_never_have"}))
    with pytest.raises(NotImplementedError, match="bionemo-ir >= 0.3.0"):
        search_from_native(_native(_strategy(export_candidates=True)))


def test_upstream_save_and_resume_keep_their_own_message() -> None:
    """The two state formats are refused separately, so a user is not told the wrong reason."""
    with pytest.raises(NotImplementedError, match="does not save or resume"):
        search_from_native(_native(_strategy(save_states=True)))
    with pytest.raises(NotImplementedError, match="does not save or resume"):
        search_from_native(_native(_strategy(resume_from="candidate.pt", resume_seed=7)))


def test_a_non_beam_strategy_is_refused() -> None:
    with pytest.raises(NotImplementedError, match="only bridges native beam search"):
        search_from_native(_native(SimpleNamespace()))


@pytest.mark.skipif(BIOIR_INSTALLED, reason="needs an environment without bionemo_ir to see the import order")
def test_refusals_are_raised_before_bioir_is_imported() -> None:
    """A bridgeable strategy gets as far as the BioIR import; an unbridgeable one never does."""
    with pytest.raises(ModuleNotFoundError, match="bionemo_ir"):
        search_from_native(_native(_strategy()))
    with pytest.raises(NotImplementedError, match="does not save or resume"):
        search_from_native(_native(_strategy(save_states=True)))


def _exported(rows: int, nsamples: int) -> dict:
    """An export whose every tensor value is its own row index, so stacking order is visible."""
    tensor = torch.arange(rows, dtype=torch.float32).reshape(rows, 1, 1)
    return {
        "candidates": {"bb_ca": tensor.clone()},
        "clean": {"bb_ca": tensor.clone() + 100},
        "step": 40,
        "example_ids": [f"input{index % nsamples}" for index in range(rows)],
        "nsamples": nsamples,
        "beam_width": rows // nsamples,
        "backend": "bioir",
    }


def test_resume_stacks_each_slot_in_request_order() -> None:
    """A slot spans every input; BioIR takes one state per slot, not one per candidate."""
    from rfproteina.nn.bioir_full import _resume_arguments

    # 2 inputs, 6 candidates; keep candidate 4 for input 0 and candidate 3 for input 1 in slot 0.
    arguments = _resume_arguments(_exported(6, 2), [4, 3, 0, 5], 2, 2, torch.device("cpu"))
    assert arguments["resume_step"] == 40
    assert [state["bb_ca"].flatten().tolist() for state in arguments["resume_states"]] == [[4, 3], [0, 5]]
    assert [state["bb_ca"].flatten().tolist() for state in arguments["resume_clean"]] == [[104, 103], [100, 105]]


def test_resume_refuses_a_selection_that_moves_a_candidate_between_inputs() -> None:
    """Mixing inputs keeps every shape valid, so nothing downstream would notice."""
    from rfproteina.nn.bioir_full import _resume_arguments

    with pytest.raises(ValueError, match="j % nsamples"):
        _resume_arguments(_exported(6, 2), [0, 2, 1, 3], 2, 2, torch.device("cpu"))
    with pytest.raises(ValueError, match="beam_width x inputs"):
        _resume_arguments(_exported(6, 2), [0, 1], 2, 2, torch.device("cpu"))
    with pytest.raises(ValueError, match=r"must lie in \[0, 6\)"):
        _resume_arguments(_exported(6, 2), [6, 1, 0, 3], 2, 2, torch.device("cpu"))


@pytest.mark.skipif(SEAM_AVAILABLE, reason="needs a bionemo-ir from before the seam")
@pytest.mark.skipif(not BIOIR_INSTALLED, reason="needs bionemo_ir installed")
def test_an_old_bioir_is_refused_by_the_real_signature() -> None:
    """The same message as the monkeypatched test, reached without monkeypatching anything."""
    with pytest.raises(NotImplementedError, match="bionemo-ir >= 0.3.0"):
        search_from_native(_native(_strategy(export_candidates=True)))
