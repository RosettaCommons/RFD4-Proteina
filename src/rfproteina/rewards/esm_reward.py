"""Frozen ESM-2 sequence embeddings and a portable linear reward head."""

import argparse
import csv
import hashlib
import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from rfproteina.rewards.reward_model import SEARCH_REWARD_KEY, RewardModel
from rfproteina.utils.common import design_output_dir
from rfproteina.utils.structure_files import find_structure_file, is_structure_file

# Official FAIR ESM-2 checkpoints, with (number of layers, embedding width).
ESM2_MODELS = {
    "esm2_t6_8M_UR50D": (6, 320),
    "esm2_t12_35M_UR50D": (12, 480),
    "esm2_t30_150M_UR50D": (30, 640),
    "esm2_t33_650M_UR50D": (33, 1280),
    "esm2_t36_3B_UR50D": (36, 2560),
    "esm2_t48_15B_UR50D": (48, 5120),
}
MAX_RESIDUES = 1022
PROTEIN_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
_METRIC_DIRS = ("inference_metrics", "evaluation_metrics")
_SAMPLE_DIRS = ("inference_samples", "evaluation_samples")


def _embedding_width(model_name: str, repr_layer: int) -> int:
    if model_name not in ESM2_MODELS:
        raise ValueError(f"Unsupported FAIR ESM-2 checkpoint: {model_name!r}.")
    num_layers, width = ESM2_MODELS[model_name]
    if type(repr_layer) is not int or not 0 <= repr_layer <= num_layers:
        raise ValueError(f"repr_layer for {model_name} must be an integer between 0 and {num_layers}.")
    return width


@dataclass(frozen=True)
class ESMLinearReward:
    """A scalar predictor on raw mean-pooled residue embeddings."""

    model_name: str
    repr_layer: int
    weights: np.ndarray
    intercept: float
    target: str

    def __post_init__(self) -> None:
        width = _embedding_width(self.model_name, self.repr_layer)
        if self.weights.shape != (width,) or not np.isfinite(self.weights).all():
            raise ValueError(f"Expected {width} finite weights for {self.model_name}.")
        if not np.isfinite(self.intercept) or not isinstance(self.target, str) or not self.target.strip():
            raise ValueError("An ESM reward needs a finite intercept and a nonempty target name.")

    @classmethod
    def load(cls, path: str | Path) -> "ESMLinearReward":
        """Load the versioned JSON head, including its encoder identity."""
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict) or data.get("version") != 1 or data.get("pooling") != "mean_residues":
            raise ValueError(f"Unsupported ESM reward format in {path}. Expected version 1 with mean_residues pooling.")
        return cls(
            model_name=data["model_name"],
            repr_layer=data["repr_layer"],
            weights=np.asarray(data["weights"], dtype=np.float64),
            intercept=float(data["intercept"]),
            target=data["target"],
        )

    def score(self, embedding: np.ndarray) -> float:
        if embedding.shape != self.weights.shape:
            raise ValueError(f"Expected ESM embedding shape {self.weights.shape}, got {embedding.shape}.")
        value = float(embedding @ self.weights + self.intercept)
        if not np.isfinite(value):
            raise ValueError("ESM reward score is nonfinite.")
        return value


class ESM2Encoder:
    """Embed a canonical protein sequence with FAIR ESM-2 in float32."""

    def __init__(self, model_name: str, repr_layer: int, device: str) -> None:
        _embedding_width(model_name, repr_layer)
        try:
            from esm import pretrained
        except ImportError as exc:
            raise ImportError("Install the rfd4-proteina[esm] extra to use an ESM reward.") from exc
        import torch

        base_url = "https://dl.fbaipublicfiles.com/fair-esm"
        # Official ESM-2 files store their configuration in argparse.Namespace.
        # Allow that type while keeping PyTorch's restricted checkpoint loader.
        with torch.serialization.safe_globals([Namespace]):
            weights = torch.hub.load_state_dict_from_url(
                f"{base_url}/models/{model_name}.pt", map_location="cpu", weights_only=True
            )
            regression = torch.hub.load_state_dict_from_url(
                f"{base_url}/regression/{model_name}-contact-regression.pt", map_location="cpu", weights_only=True
            )
        model, alphabet = pretrained.load_model_and_alphabet_core(model_name, weights, regression)
        self.model = model.to(device=device, dtype=torch.float32).eval()
        self.batch_converter = alphabet.get_batch_converter()
        self.device = torch.device(device)
        self.repr_layer = repr_layer

    def embed(self, sequence: str) -> np.ndarray:
        import torch

        if not sequence or len(sequence) > MAX_RESIDUES or set(sequence) - PROTEIN_ALPHABET:
            raise ValueError(f"ESM reward requires 1-{MAX_RESIDUES} canonical amino acids.")
        _, _, tokens = self.batch_converter([("candidate", sequence)])
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=False):
            outputs = self.model(tokens.to(self.device), repr_layers=[self.repr_layer], return_contacts=False)
            # Token 0 is BOS, and EOS follows the last residue. Exclude both.
            residues = outputs["representations"][self.repr_layer][0, 1 : len(sequence) + 1]
            return residues.mean(dim=0).cpu().numpy()


def fit_ridge(
    embeddings: np.ndarray, targets: np.ndarray, *, model_name: str, repr_layer: int, target: str, alpha: float
) -> ESMLinearReward:
    """Fit standardized ridge and fold its preprocessing into the exported coefficients."""
    width = _embedding_width(model_name, repr_layer)
    x = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != width or len(x) < 2 or y.shape != (len(x),):
        raise ValueError(f"Expected at least two aligned embeddings of width {width} and one target per row.")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge training needs finite embeddings, finite targets and positive finite alpha.")

    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-10] = 1.0
    standardized = (x - center) / scale
    target_mean = y.mean()
    penalty = alpha * width
    if len(x) < width:
        kernel = standardized @ standardized.T + penalty * np.eye(len(x))
        weights = standardized.T @ np.linalg.solve(kernel, y - target_mean)
    else:
        gram = standardized.T @ standardized + penalty * np.eye(width)
        weights = np.linalg.solve(gram, standardized.T @ (y - target_mean))
    weights /= scale
    return ESMLinearReward(model_name, repr_layer, weights, float(target_mean - center @ weights), target)


def _find_run_scores(run_dir: Path) -> Path:
    candidates = [run_dir / name / "per_example_metrics.csv" for name in _METRIC_DIRS]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        reason = "multiple tables" if matches else "no consolidated per-example metrics table"
        raise ValueError(f"Found {reason} in {run_dir}. Select the per-design CSV explicitly with --scores.")
    return matches[0]


def _resolve_design_path(recorded: str, scores_path: Path, run_dir: Path | None, example_id: str | None) -> Path:
    """Resolve the table's source path, including a preserved sample subtree in a moved run."""
    if not is_structure_file(recorded, include_pdb=True):
        raise ValueError(f"Invalid designed_structure_path {recorded!r} in {scores_path}.")
    path = Path(recorded)
    if path.is_file():
        return path.resolve()
    alternatives = [] if path.is_absolute() else [scores_path.parent / path]
    if run_dir is not None:
        if not path.is_absolute():
            alternatives.append(run_dir / path)
        for index, component in enumerate(path.parts):
            if component in _SAMPLE_DIRS:
                alternatives.append(run_dir.joinpath(*path.parts[index:]))
        if example_id and scores_path.parent.name == "evaluation_metrics":
            alternatives.append(design_output_dir(run_dir / "evaluation_samples", example_id) / path.name)
    matches = {
        found.resolve()
        for candidate in [path, *alternatives]
        if (found := candidate if candidate.is_file() else find_structure_file(candidate)) is not None
    }
    if len(matches) == 1:
        return matches.pop()
    if matches:
        raise ValueError(f"Ambiguous designed_structure_path {recorded!r} in {scores_path}: {sorted(matches)}.")
    raise FileNotFoundError(f"Cannot find design structure {recorded!r} from {scores_path}.")


def _load_reward_terms(path: Path, checkpoint: int | None) -> dict:
    """Read an existing search checkpoint or a standalone reward mapping."""
    config = OmegaConf.load(path)
    if not isinstance(config, DictConfig):
        raise ValueError(f"Expected a reward mapping in {path}.")
    if checkpoint is not None:
        search = config.get("search")
        checkpoints = search.get("checkpoints") if isinstance(search, DictConfig) else None
        matches = (
            [key for key in checkpoints if str(key) == str(checkpoint)] if isinstance(checkpoints, DictConfig) else []
        )
        if len(matches) != 1:
            raise ValueError(f"Expected one search checkpoint {checkpoint} in {path}.")
        entry = checkpoints[matches[0]]
        terms = entry.get("rewards") if isinstance(entry, DictConfig) else None
    else:
        if "search" in config:
            raise ValueError("Use --checkpoint to select the rewards from a search config.")
        terms = config.get("rewards", config)
    if not isinstance(terms, DictConfig):
        raise ValueError(f"Expected a rewards mapping in {path}.")
    return OmegaConf.to_container(terms, resolve=True, throw_on_missing=True)


def _read_scores(
    scores_path: Path, target: str, identifier: str, reward_model: RewardModel | None
) -> list[tuple[dict[str, str], float]]:
    columns = [target] if reward_model is None else [key for key, weight in reward_model.weights.items() if weight]
    rows = []
    with scores_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {identifier, *columns} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Score CSV {scores_path} is missing columns: {sorted(missing)}.")
        for row in reader:
            try:
                values = {key: float(row[key] or "nan") for key in columns}
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric score in {scores_path} at CSV line {reader.line_num}: {exc}"
                ) from exc
            if not all(np.isfinite(value) for value in values.values()):
                value = float("nan")
            elif reward_model is not None:
                value = reward_model.aggregate_metrics_output(values)
            else:
                value = values[target]
            rows.append((row, value))
    return rows


def _fit_from_embeddings(
    args: argparse.Namespace, scores_path: Path, reward_model: RewardModel | None
) -> tuple[ESMLinearReward, dict, int]:
    with np.load(args.embeddings, allow_pickle=False) as data:
        ids = data["candidate_ids"]
        embeddings = data["embeddings"]
        model_name = str(data["embedding_model"].item())
        repr_layer = data["embedding_layer"].item()
    if ids.ndim != 1 or ids.dtype.kind != "U" or len(set(ids)) != len(ids):
        raise ValueError("NPZ candidate_ids must be a unique Unicode vector aligned with the embeddings.")
    if embeddings.ndim != 2 or len(embeddings) != len(ids):
        raise ValueError("NPZ embeddings must have one row per candidate_id.")

    scores = {}
    for row, value in _read_scores(scores_path, args.target, "candidate_id", reward_model):
        candidate_id = row["candidate_id"]
        if candidate_id in scores:
            raise ValueError(f"Duplicate candidate_id in score CSV: {candidate_id!r}.")
        scores[candidate_id] = value
    targets = np.array([scores.get(candidate_id, np.nan) for candidate_id in ids])
    valid = np.isfinite(targets)
    head = fit_ridge(
        embeddings[valid],
        targets[valid],
        model_name=model_name,
        repr_layer=repr_layer,
        target=args.target,
        alpha=args.alpha,
    )
    return head, {"n_candidates": int(valid.sum())}, len(ids)


def _fit_from_run(
    args: argparse.Namespace, scores_path: Path, reward_model: RewardModel | None
) -> tuple[ESMLinearReward, dict, int]:
    from atomworks.io.parser import parse

    from rfproteina.utils.chain import get_chain_of_interest, get_protein_chain_sequence
    from rfproteina.utils.parse_presets import ANNOTATED_CIF

    run_dir = args.run_dir
    if run_dir is None and scores_path.parent.name in _METRIC_DIRS:
        run_dir = scores_path.parent.parent
    model_name = args.model or "esm2_t33_650M_UR50D"
    layer = args.layer if args.layer is not None else ESM2_MODELS[model_name][0]
    _embedding_width(model_name, layer)
    device = args.device or "cuda"
    rows = _read_scores(scores_path, args.target, "designed_structure_path", reward_model)
    labeled = [(row, value) for row, value in rows if np.isfinite(value)]
    if len(labeled) < 2:
        raise ValueError(f"Need at least two designs with finite {args.target!r} values in {scores_path}.")

    sequences, targets, seen = [], [], set()
    for row, value in labeled:
        path = _resolve_design_path(row["designed_structure_path"] or "", scores_path, run_dir, row.get("example_id"))
        if path in seen:
            raise ValueError(f"Duplicate design structure {path} in {scores_path}. Use one score row per design.")
        seen.add(path)
        try:
            atoms = parse(path, config=ANNOTATED_CIF)["assemblies"]["1"][0]
            chain_id = args.chain_id or row.get("chain_of_interest")
            if not chain_id:
                chain_id, unambiguous = get_chain_of_interest(atoms)
                if not unambiguous:
                    chain_id = None
            sequence = get_protein_chain_sequence(atoms, chain_id)
            if not 1 <= len(sequence) <= MAX_RESIDUES:
                raise ValueError(f"Expected a chain of 1-{MAX_RESIDUES} residues, got {len(sequence)}.")
        except Exception as exc:
            raise ValueError(f"Cannot extract the ESM chain from {path}: {exc}") from exc
        sequences.append(sequence)
        targets.append(value)

    encoder = ESM2Encoder(model_name, layer, device)
    unique_sequences = dict.fromkeys(sequences)
    print(f"Embedding {len(unique_sequences)} unique sequences from {len(sequences)} labeled designs.", flush=True)
    embeddings = {sequence: encoder.embed(sequence) for sequence in unique_sequences}
    head = fit_ridge(
        np.stack([embeddings[sequence] for sequence in sequences]),
        np.asarray(targets),
        model_name=model_name,
        repr_layer=layer,
        target=args.target,
        alpha=args.alpha,
    )
    return (
        head,
        {
            "n_candidates": len(sequences),
            "n_unique_sequences": len(unique_sequences),
            "chain_id": args.chain_id,
            "device": device,
        },
        len(rows),
    )


def main() -> None:
    """Fit an ESM reward head from a scored RFD4 run or precomputed embeddings."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run-dir", type=Path, help="RFD4 run directory containing per-example metrics")
    source.add_argument("--embeddings", type=Path, help="Candidate-indexed ESM-2 NPZ")
    parser.add_argument("--scores", type=Path, help="Per-design CSV. Overrides automatic discovery in --run-dir.")
    objective = parser.add_mutually_exclusive_group(required=True)
    objective.add_argument("--target", help="Numeric score column to predict")
    objective.add_argument("--rewards", type=Path, help="YAML reward terms or a search config with --checkpoint")
    parser.add_argument("--checkpoint", type=int, help="Checkpoint whose rewards to read from a search config")
    parser.add_argument("--model", choices=ESM2_MODELS, help="Run mode ESM-2 checkpoint (default: 650M)")
    parser.add_argument("--layer", type=int, help="Run mode representation layer (default: last layer)")
    parser.add_argument("--device", help="Run mode embedding device (default: cuda)")
    parser.add_argument("--chain-id", help="Run mode override for the chain of interest (IID when available)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge penalty divided by the embedding width")
    parser.add_argument("--output", type=Path, required=True, help="New JSON reward head")
    args = parser.parse_args()
    if args.scores is None and args.run_dir is None:
        parser.error("Provide --run-dir or --scores. --embeddings requires --scores.")
    if args.checkpoint is not None and (args.rewards is None or args.checkpoint <= 0):
        parser.error("--checkpoint must be a positive step and requires --rewards.")
    if args.embeddings is not None and any(
        option is not None for option in (args.model, args.layer, args.device, args.chain_id)
    ):
        parser.error("--model, --layer, --device and --chain-id apply only when embedding structures from a run.")
    if not np.isfinite(args.alpha) or args.alpha <= 0:
        parser.error("--alpha must be positive and finite.")
    if args.output.exists():
        raise FileExistsError(f"Reward head already exists: {args.output}")
    if not args.output.parent.is_dir():
        raise FileNotFoundError(f"Reward head output directory does not exist: {args.output.parent}")

    scores_path = (args.scores if args.scores is not None else _find_run_scores(args.run_dir)).resolve()
    with scores_path.open("rb") as handle:
        scores_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    reward_model = None
    objective_info = {}
    if args.rewards is not None:
        terms = _load_reward_terms(args.rewards, args.checkpoint)
        reward_model = RewardModel(terms)
        if not any(reward_model.weights.values()):
            raise ValueError("A training reward needs at least one nonzero term.")
        args.target = SEARCH_REWARD_KEY
        objective_info = {"rewards": terms, "rewards_path": str(args.rewards.resolve()), "checkpoint": args.checkpoint}
    fit = _fit_from_embeddings if args.embeddings is not None else _fit_from_run
    head, fit_info, n_total = fit(args, scores_path, reward_model)
    artifact = {
        "version": 1,
        "pooling": "mean_residues",
        "model_name": head.model_name,
        "repr_layer": head.repr_layer,
        "target": head.target,
        "weights": head.weights.tolist(),
        "intercept": head.intercept,
        "fit": {
            "alpha": args.alpha,
            "scores_path": str(scores_path),
            "scores_sha256": scores_hash,
            **fit_info,
            **objective_info,
        },
    }
    with args.output.open("x") as handle:
        json.dump(artifact, handle, indent=2, allow_nan=False)
        handle.write("\n")
    n_labeled = fit_info["n_candidates"]
    print(f"Wrote {args.output} using {n_labeled}/{n_total} candidates with finite {args.target} values.")


if __name__ == "__main__":
    main()
