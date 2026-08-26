import pytest

from rfproteina.utils.env import load_dotenv


def test_load_dotenv_raises_when_cluster_unset(monkeypatch):
    """Regression: an unset CLUSTER used to silently resolve to 'digs' rather than raising."""
    monkeypatch.delenv("CLUSTER", raising=False)
    with pytest.raises(ValueError, match="CLUSTER environment variable is not set"):
        load_dotenv()


def test_load_dotenv_raises_for_unknown_cluster(monkeypatch):
    monkeypatch.setenv("CLUSTER", "not-a-real-cluster")
    with pytest.raises(ValueError, match="No .env file found for cluster"):
        load_dotenv()
