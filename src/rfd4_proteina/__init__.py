"""Public RFD4-Proteina API: ``import rfd4_proteina as rfd4``."""

from collections.abc import Mapping, Sequence
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rfproteina.proteina import Proteina
    from rfproteina.symmetry.point_groups import SymmetryConfig, parse_symmetry

__version__ = version("rfd4-proteina")
__all__ = ["Proteina", "SymmetryConfig", "__version__", "generate", "parse_symmetry"]


def __getattr__(name: str) -> Any:
    """Load the model on demand to avoid initializing Torch during package inspection."""
    if name == "Proteina":
        from rfproteina.proteina import Proteina

        globals()[name] = Proteina
        return Proteina
    if name in ("SymmetryConfig", "parse_symmetry"):
        from rfproteina.symmetry import point_groups

        attr = getattr(point_groups, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def generate(
    model: "Proteina | None" = None,
    *,
    symmetry: "SymmetryConfig | Mapping[str, Any] | str | None" = None,
    sym_radius: float | None = None,
    overrides: "Sequence[str]" = (),
) -> None:
    """Compose ``inference.yaml`` with Hydra ``overrides`` and run generation.

    Args:
        model: A model loaded with ``Proteina.from_pretrained``; ``None`` loads the checkpoint named by the config.
        symmetry: Optional point-group symmetry specification (e.g. ``"C3"``, ``"D2"``, ``"T"``, ``"O"``, ``"I"``,
            a mapping, or a :class:`~rfproteina.symmetry.point_groups.SymmetryConfig`).
        sym_radius: Optional ASU radial offset in Angstroms from the symmetry origin.
        overrides: Hydra overrides, the same ``key=value`` strings that ``rfd4-generate`` accepts, for example
            ``["inputs=design.yaml", "out_dir=runs/first"]``.
    """
    import rootutils
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    import rfproteina
    from rfproteina.generate import run_generation
    from rfproteina.symmetry.transforms import merge_symmetry_config

    config_dir = rootutils.find_root(rfproteina.__file__, indicator=".project-root") / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="inference", overrides=list(overrides))

    if symmetry is not None or sym_radius is not None:
        base_sym = OmegaConf.select(cfg, "generation.symmetry", default=None)
        if OmegaConf.is_config(base_sym):
            base_sym = OmegaConf.to_container(base_sym, resolve=True)
        merged_sym = merge_symmetry_config(
            base_sym,
            cfg.get("symmetry") if symmetry is None else symmetry,
            sym_radius=cfg.get("sym_radius") if sym_radius is None else sym_radius,
        )
        with open_dict(cfg):
            cfg.symmetry = merged_sym.group
            if sym_radius is not None:
                cfg.sym_radius = float(sym_radius)
        with open_dict(cfg.generation):
            cfg.generation.symmetry = merged_sym.to_dict()

    run_generation(cfg, model=model)
