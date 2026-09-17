import os
import sys

root = os.path.abspath(".")
sys.path.insert(0, root)  # Adds project's root directory
# isort: split

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import hydra  # noqa: E402
import lightning as L  # noqa: E402

# import matplotlib.pyplot as plt
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from rfproteina.model_masks import compute_ae_supervision_mask  # noqa: E402
from rfproteina.partial_autoencoder.autoencoder import AutoEncoder  # noqa: E402
from rfproteina.utils.ddp import RankedLogger  # noqa: E402
from rfproteina.utils.env import load_dotenv  # noqa: E402

logger = RankedLogger(__name__, rank_zero_only=True)


def parse_args_and_cfg() -> tuple[argparse.Namespace, DictConfig, str]:
    """Return CLI arguments, the composed Hydra config, and its name."""
    parser = argparse.ArgumentParser(description="Job info")
    parser.add_argument(
        "--config_name",
        type=str,
        default="inference_ae",
        help="Name of the config yaml file.",
    )
    parser.add_argument("--config_number", type=int, default=-1, help="Number of the config yaml file.")
    parser.add_argument(
        "--config_subdir",
        type=str,
        help="(Optional) Name of directory with config files, if not included uses base inference config.\
            Likely only used when submitting to the cluster with script.",
    )
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parents[3] / "configs"
    if args.config_subdir is not None:
        config_path = config_path / args.config_subdir

    with hydra.initialize_config_dir(str(config_path), version_base=hydra.__version__):
        # If number provided use it, otherwise name
        if args.config_number != -1:
            config_name = f"inf_{args.config_number}"
        else:
            config_name = args.config_name
        cfg = hydra.compose(config_name=config_name)
        logger.info(f"Inference config {cfg}")

    return args, cfg, config_name


def extract_ckpt_info(ckpt_file_path: str) -> tuple[str, str]:
    ae_name = ckpt_file_path.split("/")[-3]
    ckpt_name = ckpt_file_path.split("/")[-1]
    return ae_name, ckpt_name


def setup(
    cfg: dict,
    config_name: str,
    create_root: bool = True,
) -> str:
    """
    Checks if metrics being computed are compatible, sets the right seed, and creates the root directory
    where the run will store things.

    Returns:
        Path of the root directory (string)
    """
    logger.info(" ".join(sys.argv))

    assert torch.cuda.is_available(), "CUDA not available"  # Needed for ESMfold and designability

    # Set root path for this inference run
    root_path = f"./inference/{config_name}"
    if create_root:
        os.makedirs(root_path, exist_ok=True)
    else:
        if not os.path.exists(root_path):
            raise ValueError(f"Results path {root_path} does not exist")

    # Set seed
    logger.info(f"Seeding everything to seed {cfg.seed}")
    L.seed_everything(cfg.seed)

    return root_path


def load_dataloader(cfg: object) -> object:
    """
    Load dataloader from dataset config using new atomworks pipeline.
    Matches training's approach EXACTLY - uses assemble_distributed_loader.
    """
    from rfproteina.utils.datasets import (
        assemble_distributed_loader,
        recursively_instantiate_datasets_and_samplers,
    )

    # Use cfg.datasets.train directly, just like training does
    # This is already composed via Hydra defaults in inference_ae.yaml
    logger.info("Loading dataset from cfg.datasets.train (matches training)")

    # Instantiate dataset (matches train.py line 91-95)
    dataset_and_sampler = recursively_instantiate_datasets_and_samplers(cfg.datasets.train)
    dataset = dataset_and_sampler["dataset"]
    sampler = dataset_and_sampler.get("sampler", None)

    logger.info(f"Dataset size: {len(dataset)}")

    # Create dataloader using assemble_distributed_loader - EXACTLY like training does
    # For inference, use rank=0 and world_size=1 (non-distributed)
    # For mixed datasets, sampler needs shuffle=True to use probability-based sampling
    # Reproducibility is still maintained via cfg.seed (set in setup())
    dataloader = assemble_distributed_loader(
        dataset=dataset,
        sampler=sampler,
        rank=0,
        world_size=1,
        n_examples_per_epoch=len(dataset),  # KEY: This is what training does!
        loader_cfg=hydra.utils.instantiate(cfg.dataloaders.val),
        shuffle=True,  # Required for MixedSampler to use probabilities!
    )

    logger.info(f"Dataloader created with {len(dataloader)} batches")

    return dataloader


def extract_pdb_ids(predictions: list[tuple]) -> list[str]:
    logger.info("Extracting example IDs")
    vals = []
    for x_in, _ in predictions:
        # New batch format uses example_id_batch (list of IDs)
        if "example_id_batch" in x_in:
            v = x_in["example_id_batch"]
        elif "id" in x_in:  # Fallback for old format
            v = x_in["id"]
        else:
            raise KeyError("Neither 'example_id_batch' nor 'id' found in batch")
        vals += v
    return vals


def compute_all_atom_rmsd(predictions: list[tuple], model: AutoEncoder) -> list[float]:
    logger.info("Computing all-atom RMSD")
    vals = []
    for x_in, x_out in predictions:
        loss_mask = compute_ae_supervision_mask(x_in)
        v = model.compute_struct_rec_loss(
            output_dec=x_out,
            batch=x_in,
            loss_mask=loss_mask,
        )["rmsd_no_align_a37_ang_justlog"]
        vals += v.tolist()
    return vals


def compute_sec_rec_rate(predictions: list[tuple], model: AutoEncoder) -> list[float]:
    logger.info("Computing sequence recovery rate")
    vals = []
    for x_in, x_out in predictions:
        loss_mask = compute_ae_supervision_mask(x_in)
        v = model.compute_seq_rec_loss(
            output_dec=x_out,
            batch=x_in,
            loss_mask=loss_mask,
        )["seq_rec_rate_justlog"]
        vals += v.tolist()
    return vals


def compute_kl_latent(predictions: list[tuple], model: AutoEncoder) -> list[float]:
    logger.info("Computing KL latent penalty")
    vals = []
    for x_in, x_out in predictions:
        v = model.compute_kl_penalty(
            mean=x_out["mean"],
            log_scale=x_out["log_scale"],
            mask=compute_ae_supervision_mask(x_in),
            w=1.0,
        )["kl_now_justlog"]
        vals += v.tolist()
    return vals


def compute_metric(metric: str, predictions: list[tuple], model: AutoEncoder) -> list[float]:
    if metric == "all_atom_rmsd":
        return compute_all_atom_rmsd(predictions, model)  # List of floats
    elif metric == "seq_rec_rate":
        return compute_sec_rec_rate(predictions, model)  # List of floats
    elif metric == "kl_latent_dist":
        return compute_kl_latent(predictions, model)  # List of floats
    else:
        raise OSError(f"Metric {metric} not implemented")


def get_df_stats(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    means = df[numeric_cols].mean()
    stds = df[numeric_cols].std()

    stats_data = {"stat_type": ["mean", "std"]}
    for col in numeric_cols:
        stats_data[col] = [means[col], stds[col]]

    return pd.DataFrame(stats_data)


def main() -> None:
    load_dotenv()

    # Load config
    args, cfg, config_name = parse_args_and_cfg()
    ae_name, ckpt_name = extract_ckpt_info(cfg.ckpt_file)

    # Some setup
    root_path = setup(cfg, create_root=True, config_name=config_name)
    df_file_store = os.path.join(root_path, f"../results_{config_name}.csv")
    df_file_store_summary = os.path.join(root_path, f"../results_{config_name}_summary.csv")

    # Get dataloader
    dataloader = load_dataloader(cfg)

    # Model
    model = AutoEncoder.load_from_checkpoint(cfg.ckpt_file)

    # Make predictions, store them together with inputs
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        limit_predict_batches=int(cfg.n_structs / cfg.bs),
    )
    predictions = trainer.predict(model, dataloader)
    # List of tuples, each tuple is (data_batch, predicted_batch)
    # and the predicted batch has all outputs from the endocer and decoder

    # Compute requested metrics
    metrics = {}
    metrics_to_compute = [k for k in cfg.metrics if cfg.metrics[k]]
    for metric in metrics_to_compute:
        metrics[metric] = compute_metric(metric=metric, predictions=predictions, model=model)

    # Extract PDB ids
    pdb_id = extract_pdb_ids(predictions)  # List of strs

    # Run latent analysis if requested
    if hasattr(cfg, "latent_analysis") and cfg.latent_analysis.get("enabled", False):
        from rfproteina.partial_autoencoder.latent_analysis import run_latent_analysis

        output_dir = cfg.latent_analysis.get("output_dir", "./latent_analysis")
        method = cfg.latent_analysis.get("projection_method", "tsne")
        n_samples = cfg.latent_analysis.get("n_samples", 10000)
        filter_residues = cfg.latent_analysis.get("filter_residues", None)

        # Get method-specific kwargs
        method_kwargs = cfg.latent_analysis.get(method, {})

        run_latent_analysis(
            predictions=predictions,
            output_dir=output_dir,
            method=method,
            n_samples=n_samples,
            filter_residues=filter_residues,
            **method_kwargs,
        )

    # Plot requested stuff
    dir_storages = {}

    # Create dataframe with results
    info_df = {"pdb_id": pdb_id}
    info_df.update(metrics)
    df = pd.DataFrame(info_df)

    # Save summary results
    col_names = ["ae_name", "ckpt_name", "dataset"]
    # Extract dataset name from config (now using datasets.train structure)
    dataset_name = (
        "mixed"
        if "afdb_monomer" in cfg.datasets.train and "na_complex" in cfg.datasets.train
        else next(iter(cfg.datasets.train.keys()))
        if hasattr(cfg.datasets, "train")
        else "unknown"
    )
    values = [ae_name, ckpt_name, dataset_name]
    for m in metrics:
        col_names += [f"{m}_mean", f"{m}_std", f"{m}_max", f"{m}_min"]
        vals_aux = np.array(metrics[m])
        values += [vals_aux.mean(), vals_aux.std(), vals_aux.max(), vals_aux.min()]
    col_names += list(dir_storages)
    values += [dir_storages[k] for k in dir_storages]
    df_summary = pd.DataFrame({col_names[i]: [values[i]] for i in range(len(col_names))})

    # Save dataframes
    df.to_csv(df_file_store, index=False)
    df_summary.to_csv(df_file_store_summary, index=False)

    # Save df
    df.to_csv(df_file_store, index=False)
    df_summary.to_csv(df_file_store_summary, index=False)
    print("Done saving dataframes")


if __name__ == "__main__":
    main()
