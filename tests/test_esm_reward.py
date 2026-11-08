"""Exercise the ESM reward through the fit command and the composed search metric."""

import csv
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from biotite.structure import AtomArray
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rfproteina.metrics import esm_reward_metric
from rfproteina.metrics.metric import MetricInputError, instantiate_metric_manager
from rfproteina.rewards import esm_reward
from rfproteina.rewards.esm_reward import ESMLinearReward, main
from rfproteina.search.beam_search import BeamSearchStrategy


def test_esm_reward_fit_joins_candidate_scores_and_skips_missing(monkeypatch, tmp_path):
    embeddings = np.zeros((4, 320))
    embeddings[:, 0] = [1, 2, 3, 4]
    np.savez(
        tmp_path / "embeddings.npz",
        embeddings=embeddings,
        candidate_ids=["A", "B", "C", "D"],
        embedding_model="esm2_t6_8M_UR50D",
        embedding_layer=6,
    )
    scores = tmp_path / "scores.csv"
    scores.write_text("candidate_id,quality\nC,30\nD,\nA,10\nB,20\n")
    output = tmp_path / "head.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fit",
            "--embeddings",
            str(tmp_path / "embeddings.npz"),
            "--scores",
            str(scores),
            "--target",
            "quality",
            "--alpha",
            "0.00001",
            "--output",
            str(output),
        ],
    )
    main()
    head = ESMLinearReward.load(output)
    assert json.loads(output.read_text())["fit"]["n_candidates"] == 3
    np.testing.assert_allclose([head.score(e) for e in embeddings[:3]], [10, 20, 30], atol=0.02)


def test_esm_reward_fit_from_checkpoint_terms(monkeypatch, tmp_path):
    embeddings = np.zeros((3, 320))
    embeddings[:, 0] = [1, 2, 3]
    np.savez(
        tmp_path / "embeddings.npz",
        embeddings=embeddings,
        candidate_ids=["A", "B", "C"],
        embedding_model="esm2_t6_8M_UR50D",
        embedding_layer=6,
    )
    scores = tmp_path / "scores.csv"
    scores.write_text("candidate_id,oracle.plddt_coi,refolded_backbone_rmsd.rmsd\nC,80,\nA,90,2\nB,40,10\n")
    config = tmp_path / "search.yaml"
    config.write_text(
        "search:\n"
        "  checkpoints:\n"
        '    "20":\n'
        "      rewards:\n"
        "        oracle.plddt_coi: {}\n"
        "    60:\n"
        "      rewards:\n"
        "        oracle.plddt_coi: {weight: 2, clip: {min: 50}, normalize_by: 100}\n"
        "        refolded_backbone_rmsd.rmsd: {weight: -1, clip: {max: 4}, normalize_by: 2}\n"
        "        unused.metric: {weight: 0}\n"
    )
    output = tmp_path / "head.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fit",
            "--embeddings",
            str(tmp_path / "embeddings.npz"),
            "--scores",
            str(scores),
            "--rewards",
            str(config),
            "--checkpoint",
            "60",
            "--alpha",
            "0.00001",
            "--output",
            str(output),
        ],
    )
    main()
    artifact = json.loads(output.read_text())
    assert artifact["target"] == "search.reward"
    assert artifact["fit"]["n_candidates"] == 2
    assert artifact["fit"]["checkpoint"] == 60
    assert artifact["fit"]["rewards"]["refolded_backbone_rmsd.rmsd"]["weight"] == -1
    head = ESMLinearReward.load(output)
    np.testing.assert_allclose([head.score(e) for e in embeddings[:2]], [0.8, -1.0], atol=0.003)


@pytest.mark.parametrize("kind", ["inference", "evaluation"])
def test_esm_reward_fit_from_scored_run(kind, monkeypatch, tmp_path):
    data_dir = Path(__file__).parent / "data"
    run = tmp_path / "moved_run"
    metrics = run / f"{kind}_metrics"
    metrics.mkdir(parents=True)
    rows = []
    for index, (fixture, score) in enumerate(
        [
            ("1a1e.cif", 30),
            ("sam_binder_generated_minimized.cif", 80),
            ("1a1e.cif", 30),
        ]
    ):
        design_id = f"design_{index}"
        sample_dir = run / f"{kind}_samples" / design_id
        sample_dir.mkdir(parents=True)
        if kind == "inference":
            structure = sample_dir / f"{design_id}_generated.cif"
            shutil.copyfile(data_dir / fixture, structure)
            recorded = tmp_path / "old_run" / structure.relative_to(run)
        else:
            if index == 2:
                shutil.copyfile(data_dir / fixture, sample_dir / fixture)
            else:
                (sample_dir / fixture).symlink_to(data_dir / fixture)
            recorded = Path("input_from_other_workdir") / fixture
        rows.append([design_id, recorded, "A_1", score])
    rows.append(["unscored", "missing.cif", "", ""])
    scores_path = metrics / "per_example_metrics.csv"
    with scores_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["example_id", "designed_structure_path", "chain_of_interest", "oracle.plddt_coi"])
        writer.writerows(rows)

    embedded = []

    class TestEncoder:
        def __init__(self, model_name, layer, device):
            assert (model_name, layer, device) == ("esm2_t6_8M_UR50D", 6, "cpu")

        def embed(self, sequence):
            embedded.append(sequence)
            return np.array([len(sequence)] + [0.0] * 319)

    monkeypatch.setattr(esm_reward, "ESM2Encoder", TestEncoder)
    output = tmp_path / "head.json"
    if kind == "inference":
        rewards = tmp_path / "rewards.yaml"
        rewards.write_text("rewards:\n  oracle.plddt_coi: {weight: 2}\n")
        objective = ["--rewards", str(rewards)]
        expected = [60, 160]
    else:
        objective = ["--target", "oracle.plddt_coi"]
        expected = [30, 80]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fit",
            "--run-dir",
            str(run),
            *objective,
            "--model",
            "esm2_t6_8M_UR50D",
            "--device",
            "cpu",
            "--alpha",
            "0.00001",
            "--output",
            str(output),
        ],
    )
    main()
    fit = json.loads(output.read_text())["fit"]
    assert fit["n_candidates"] == 3 and fit["n_unique_sequences"] == 2
    assert [len(s) for s in embedded] == [107, 150]
    head = ESMLinearReward.load(output)
    np.testing.assert_allclose([head.score(np.array([n] + [0.0] * 319)) for n in (107, 150)], expected, atol=0.1)


def test_esm_beam_reward_uses_selected_chain_and_caches_score(monkeypatch, tmp_path):
    head = tmp_path / "head.json"
    head.write_text(
        json.dumps(
            {
                "version": 1,
                "pooling": "mean_residues",
                "model_name": "esm2_t6_8M_UR50D",
                "repr_layer": 6,
                "target": "quality",
                "intercept": 2.0,
                "weights": [1.0] + [0.0] * 319,
            }
        )
    )
    embedded = []

    class TestEncoder:
        def __init__(self, model_name, repr_layer, device):
            assert (model_name, repr_layer, device) == ("esm2_t6_8M_UR50D", 6, "cpu")

        def embed(self, sequence):
            embedded.append(sequence)
            return np.arange(1, 321, dtype=float)

    monkeypatch.setattr(esm_reward_metric, "ESM2Encoder", TestEncoder)
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="inference",
            overrides=[
                "inference_experiment=inference_esm_beam",
                "inputs=unused.cif",
                f"metrics.inline_metrics.esm_reward.artifact_path={head}",
                "metrics.inline_metrics.esm_reward.device=cpu",
            ],
        )
    manager = instantiate_metric_manager({"esm_reward": cfg.metrics.inline_metrics.esm_reward})
    strategy = BeamSearchStrategy(**OmegaConf.to_container(cfg.search, resolve=True))
    strategy.configure_metrics(SimpleNamespace(design_metric_manager=manager))
    manager.on_validation_epoch_start(val_dataset_tags=[set()], metric_names=strategy.metric_names)

    atoms = AtomArray(4)
    atoms.chain_id = ["A", "A", "A", "B"]
    atoms.res_id = [1, 1, 2, 1]
    atoms.res_name = ["ALA", "ALA", "GLY", "ATP"]
    inputs = {"predicted_atom_array": atoms, "extra_info": {"chain_of_interest": "A", "metrics_tags": []}}
    for _ in range(2):
        result = manager(metric_names=["esm_reward"], **inputs)
        assert strategy.reward_models[40].aggregate_metrics_output(result) == 3.0
    assert embedded == ["AG"]
    with pytest.raises(MetricInputError, match="canonical amino acids"):
        manager.metrics["esm_reward"].compute(atoms, chain_of_interest="B")
    manager.on_validation_epoch_end()
