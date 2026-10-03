"""Load local settings or an explicitly selected runtime profile."""

import os
from pathlib import Path
from typing import Union

import rootutils
from dotenv import load_dotenv as _load_dotenv

PROJECT_ROOT_PATH = Path(rootutils.find_root(indicator=".project-root", search_from=__file__))
# Wheels carry their configs under rfproteina; local settings belong in the working directory.
DOTENV_ROOT_PATH = (
    Path.cwd() if Path(__file__).resolve().parents[1] == PROJECT_ROOT_PATH else PROJECT_ROOT_PATH
) / ".ipd"


def load_dotenv(
    dotenv_path: Union[str, "os.PathLike[str]"] | None = None,
    override: bool = False,
    **kwargs,
) -> None:
    """Load a public environment file or a named private cluster profile.

    If ``dotenv_path`` is provided it is used directly. Otherwise, the ``CLUSTER``
    environment variable selects the private profile when present, otherwise
    root ``.env.<cluster>``. Public installs default to ``.env.local``, falling
    back to ``.env``. Private checkouts require ``CLUSTER``; ``local`` explicitly
    selects public settings. Existing variables win unless overridden.

    Args:
        dotenv_path: Explicit path to a .env file. Skips cluster detection when given.
        override: Whether to override variables already present in the environment.
        **kwargs: Additional keyword arguments forwarded to :py:func:`dotenv.load_dotenv`.

    Raises:
        ValueError: If an explicitly selected private profile does not exist.
    """
    if dotenv_path is None:
        os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT_PATH))
        cluster = os.environ.get("CLUSTER")
        if cluster is None and DOTENV_ROOT_PATH.is_dir():
            raise ValueError(
                "CLUSTER environment variable is not set. Set it before importing rfproteina, "
                "Use `export CLUSTER=local` for a public installation."
            )
        if not cluster or cluster == "local":
            dotenv_path = DOTENV_ROOT_PATH.parent / ".env.local"
            if not dotenv_path.exists():
                dotenv_path = DOTENV_ROOT_PATH.parent / ".env"
            _load_dotenv(dotenv_path, override=override, **kwargs)
            return

        dotenv_path = DOTENV_ROOT_PATH / f".env.{cluster}"
        if not dotenv_path.exists():
            dotenv_path = DOTENV_ROOT_PATH.parent / f".env.{cluster}"

        if not dotenv_path.exists():
            raise ValueError(
                f"No .env file found for cluster '{cluster}' (looked at {dotenv_path}). "
                f"Create a root .env.{cluster} profile or select CLUSTER=local."
            )

    _load_dotenv(dotenv_path, override=override, **kwargs)
