"""A patch over load_dotenv to take care of cluster-specific .env files for training on different clusters."""

import os
from pathlib import Path
from typing import Union

import rootutils
from dotenv import load_dotenv as _load_dotenv

DOTENV_ROOT_PATH = Path(rootutils.find_root(indicator=".project-root", search_from=__file__)) / ".ipd"


def load_dotenv(
    dotenv_path: Union[str, "os.PathLike[str]"] | None = None,
    override: bool = False,
    **kwargs,
) -> None:
    """Load the cluster-specific .env file based on the CLUSTER environment variable.

    If ``dotenv_path`` is provided it is used directly. Otherwise, the ``CLUSTER``
    environment variable selects the appropriate file from ``.ipd/``.

    Args:
        dotenv_path: Explicit path to a .env file. Skips cluster detection when given.
        override: Whether to override variables already present in the environment.
        **kwargs: Additional keyword arguments forwarded to :py:func:`dotenv.load_dotenv`.

    Raises:
        ValueError: If ``CLUSTER`` is unset, or no ``.env`` file exists for the resolved
          cluster name.
    """
    if dotenv_path is None:
        cluster = os.environ.get("CLUSTER")
        if cluster is None:
            raise ValueError(
                "CLUSTER environment variable is not set. Set it before importing rfproteina, "
                "e.g. `export CLUSTER=local` (or 'digs'/'ord'/'gb200'/'coreweave' on a named "
                "cluster) in your ~/.bashrc, ~/.zshrc, or the current shell."
            )

        dotenv_path = DOTENV_ROOT_PATH / f".env.{cluster}"

        if not dotenv_path.exists():
            raise ValueError(
                f"No .env file found for cluster '{cluster}' (looked at {dotenv_path}). "
                "Check that CLUSTER is set correctly in your ~/.bashrc or ~/.zshrc."
            )

    _load_dotenv(dotenv_path, override=override, **kwargs)
