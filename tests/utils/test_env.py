import pytest

from rfproteina.utils.env import load_dotenv


def test_load_dotenv_without_cluster_loads_optional_local_env(monkeypatch, tmp_path):
    """A public installation needs no private runtime profile."""
    import os

    from rfproteina.utils import env

    monkeypatch.delenv("CLUSTER", raising=False)
    monkeypatch.delenv("RFD4_TEST_ENV", raising=False)
    monkeypatch.setattr(env, "DOTENV_ROOT_PATH", tmp_path / ".ipd")
    load_dotenv()
    assert "RFD4_TEST_ENV" not in os.environ
    (tmp_path / ".env").write_text("RFD4_TEST_ENV=local\n")
    load_dotenv()
    assert os.environ["RFD4_TEST_ENV"] == "local"


def test_load_dotenv_raises_for_unknown_cluster(monkeypatch):
    monkeypatch.setenv("CLUSTER", "not-a-real-cluster")
    with pytest.raises(ValueError, match="No .env file found for cluster"):
        load_dotenv()
