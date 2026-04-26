from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Protocol


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class EmbedderProtocol(Protocol):
    def embed_query(self, text: str) -> list[float]:
        ...


def tokenize_text(text: str) -> list[str]:
    """Tokenize deck text into stable alphanumeric terms for bag-of-words comparisons."""
    return [token.lower() for token in TOKEN_RE.findall(text)]


def build_term_counter(text: str) -> Counter[str]:
    """Convert text into a frequency counter so lexical overlap can be scored consistently."""
    return Counter(tokenize_text(text))


def cosine_similarity_from_counters(left: Counter[str], right: Counter[str]) -> float:
    """Score lexical similarity between two token counters on a 0..1 cosine scale."""
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def cosine_similarity_from_vectors(left: list[float], right: list[float]) -> float:
    """Score semantic similarity between two dense embeddings on a 0..1 cosine scale."""
    numerator = sum(x * y for x, y in zip(left, right))
    left_norm = math.sqrt(sum(x * x for x in left))
    right_norm = math.sqrt(sum(y * y for y in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def token_cosine_similarity(left_text: str, right_text: str) -> float:
    """Convenience wrapper for lexical cosine similarity over raw text inputs."""
    return cosine_similarity_from_counters(build_term_counter(left_text), build_term_counter(right_text))


def embedding_cosine_similarity(left_text: str, right_text: str, embedder: EmbedderProtocol) -> float:
    """Convenience wrapper for semantic cosine similarity over raw text inputs."""
    return cosine_similarity_from_vectors(
        embedder.embed_query(left_text),
        embedder.embed_query(right_text),
    )


def mean_score(values: Iterable[float]) -> float:
    """Average a finite sequence of metric scores and return 0 for empty inputs."""
    values_list = list(values)
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)


def normalize_dtw_cost(cost: float) -> float:
    """Map an unbounded DTW alignment cost into a bounded 0..1 similarity score."""
    return 1.0 / (1.0 + cost)
