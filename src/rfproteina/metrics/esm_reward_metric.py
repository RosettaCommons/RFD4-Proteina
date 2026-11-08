"""Sequence-based reward for decoded RFD4 candidates."""

from collections import OrderedDict

from biotite.structure import AtomArray

from rfproteina.metrics.metric import Metric, MetricInputError
from rfproteina.rewards.esm_reward import ESM2Encoder, ESMLinearReward
from rfproteina.utils.chain import get_protein_chain_sequence


class ESMRewardMetric(Metric):
    """Score the full sequence of one canonical protein chain with a trained ESM-2 head.

    By default the metric uses RFD4's unambiguous chain of interest. Set
    ``chain_id`` to choose a chain explicitly (``chain_iid`` when present).
    """

    def __init__(
        self, artifact_path: str, device: str = "cuda", chain_id: str | None = None, cache_size: int = 4096, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("ESM reward cache_size must be a nonnegative integer.")
        self.artifact_path = artifact_path
        self.device = device
        self.chain_id = chain_id
        self.cache_size = cache_size
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._head: ESMLinearReward | None = None
        self._encoder: ESM2Encoder | None = None

    @property
    def kwargs_to_compute_args(self) -> dict[str, tuple[str, ...]]:
        return {
            "predicted_atom_array": ("predicted_atom_array",),
            "chain_of_interest": ("extra_info", "chain_of_interest"),
        }

    @property
    def optional_kwargs(self) -> frozenset[str]:
        return frozenset({"chain_of_interest"})

    def setup(self) -> None:
        if self._encoder is None:
            head = ESMLinearReward.load(self.artifact_path)
            self._encoder = ESM2Encoder(head.model_name, head.repr_layer, self.device)
            self._head = head

    def teardown(self) -> None:
        self._cache.clear()
        self._encoder = None
        self._head = None

    def compute(self, predicted_atom_array: AtomArray, chain_of_interest: str | None = None) -> dict[str, float]:
        chain_id = self.chain_id if self.chain_id is not None else chain_of_interest
        try:
            sequence = get_protein_chain_sequence(predicted_atom_array, chain_id)
        except ValueError as exc:
            raise MetricInputError(str(exc)) from exc

        if sequence in self._cache:
            self._cache.move_to_end(sequence)
            return {"score": self._cache[sequence]}
        self.setup()
        assert self._head is not None and self._encoder is not None
        score = self._head.score(self._encoder.embed(sequence))
        if self.cache_size:
            self._cache[sequence] = score
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return {"score": score}
