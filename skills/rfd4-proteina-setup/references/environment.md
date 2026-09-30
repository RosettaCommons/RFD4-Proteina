# Environment

The public setup creates a Python 3.12 virtualenv and installs the project with
its dev extra. AtomWorks is a Git dependency pinned in pyproject.toml to a public
revision of release/atomworks-3-0. It is not a submodule.

Activate .venv before invoking Python entry points. DATA_PATH identifies the
downloaded checkpoint root; it may be exported or saved in the optional root
.env file. CLUSTER is optional and only selects a user-created runtime profile.

For the supported commands, see [installation](../../../../docs/installation.rst).
For RDKit native-library issues with uv Python, the Makefile test target
preloads the system libstdc++ as a workaround.
