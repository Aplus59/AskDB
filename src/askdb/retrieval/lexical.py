"""BM25 ranking over table documents.

This is the baseline schema retriever. It has no model dependency and runs in
microseconds, which makes it the thing any embedding-based retriever has to
beat before the added cost and latency are worth paying for.
"""

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from askdb.retrieval.documents import TableDocument, tokenize

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


@dataclass(frozen=True)
class Ranked:
    table: str
    score: float


class BM25:
    """Standard BM25 over a fixed set of table documents."""

    def __init__(
        self,
        documents: Iterable[TableDocument],
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        self._documents = tuple(documents)
        self._k1 = k1
        self._b = b

        self._frequencies = [Counter(document.tokens) for document in self._documents]
        self._lengths = [len(document) for document in self._documents]

        total_length = sum(self._lengths)
        self._average_length = total_length / len(self._documents) if self._documents else 0.0

        self._document_frequency: Counter[str] = Counter()
        for frequency in self._frequencies:
            self._document_frequency.update(frequency.keys())

    def __len__(self) -> int:
        return len(self._documents)

    def _inverse_document_frequency(self, token: str) -> float:
        total = len(self._documents)
        containing = self._document_frequency.get(token, 0)
        return math.log(1 + (total - containing + 0.5) / (containing + 0.5))

    def _score_document(self, index: int, query_tokens: list[str]) -> float:
        frequency = self._frequencies[index]
        length = self._lengths[index]

        # An average length of zero means every document is empty, in which
        # case normalisation has nothing to normalise against.
        if self._average_length > 0:
            normalisation = 1 - self._b + self._b * (length / self._average_length)
        else:
            normalisation = 1.0

        score = 0.0
        for token in query_tokens:
            occurrences = frequency.get(token, 0)
            if occurrences == 0:
                continue
            numerator = occurrences * (self._k1 + 1)
            denominator = occurrences + self._k1 * normalisation
            score += self._inverse_document_frequency(token) * (numerator / denominator)
        return score

    def search(self, question: str, limit: int | None = None) -> list[Ranked]:
        """Rank tables against a question, best first.

        Ties are broken by table name so that repeated runs produce identical
        output; an evaluation that shuffles under ties cannot be compared with
        itself.
        """
        query_tokens = tokenize(question)
        ranked = [
            Ranked(table=document.table, score=self._score_document(index, query_tokens))
            for index, document in enumerate(self._documents)
        ]
        ranked.sort(key=lambda item: (-item.score, item.table))
        return ranked[:limit] if limit is not None else ranked
