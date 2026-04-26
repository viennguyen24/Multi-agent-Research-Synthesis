from __future__ import annotations

from pydantic import BaseModel

from eval.metrics.benchmark.deck_bench.math_utils import (
    EmbedderProtocol,
    embedding_cosine_similarity,
    mean_score,
    normalize_dtw_cost,
    token_cosine_similarity,
)
from eval.metrics.benchmark.deck_bench.prompts import coherence_prompt, coherence_system_prompt
from eval.schema import DeckView
from src.llm.llm import LLMConfig, get_llm, init_from_config


class DeckJudgeResponse(BaseModel):
    reason: str
    score: int


def deck_faithfulness_metric(
    generated_deck: DeckView,
    source_chunks: list[str],
    embedder: EmbedderProtocol,
) -> float:
    """Measure whether the whole generated deck stays semantically close to its source material.

    Internally this metric concatenates all generated slide text, concatenates all source chunks, embeds
    both texts with the shared sentence-transformer model, and computes cosine similarity. The effect is
    to reward decks that preserve the paper's substance across the full presentation rather than only on
    one or two slides.
    """
    return embedding_cosine_similarity(
        "\n".join(slide.combined_text for slide in generated_deck.slides),
        "\n".join(source_chunks),
        embedder,
    )


def deck_fidelity_metric(
    generated_deck: DeckView,
    reference_deck: DeckView,
    embedder: EmbedderProtocol,
) -> float:
    """Measure semantic agreement between the full generated deck and the full reference deck.

    Internally this metric concatenates each deck into a single text stream, embeds both streams, and
    scores cosine similarity. The effect is to reward overall content coverage and semantic match against
    the target benchmark deck without depending on one-to-one slide boundaries.
    """
    return embedding_cosine_similarity(
        "\n".join(slide.combined_text for slide in generated_deck.slides),
        "\n".join(slide.combined_text for slide in reference_deck.slides),
        embedder,
    )


def transition_similarity_metric(generated_deck: DeckView, reference_deck: DeckView) -> float:
    """Measure whether adjacent generated slides transition like adjacent reference slides.

    Internally this metric scores lexical cosine similarity between corresponding two-slide windows
    `(slide_i + slide_i+1)` across the generated and reference decks, then averages those scores. The
    effect is to reward narrative continuity and ordering, not just per-slide topical overlap.
    """
    generated_pairs = list(zip(generated_deck.slides, generated_deck.slides[1:]))
    reference_pairs = list(zip(reference_deck.slides, reference_deck.slides[1:]))
    if not generated_pairs or not reference_pairs:
        return 0.0
    limit = min(len(generated_pairs), len(reference_pairs))
    return mean_score(
        token_cosine_similarity(
            f"{generated_pairs[index][0].combined_text}\n{generated_pairs[index][1].combined_text}",
            f"{reference_pairs[index][0].combined_text}\n{reference_pairs[index][1].combined_text}",
        )
        for index in range(limit)
    )


def dtw_sequence_similarity_metric(
    generated_deck: DeckView,
    reference_deck: DeckView,
) -> tuple[float, list[tuple[int, int]]]:
    """Measure sequence alignment quality between generated and reference slide orders.

    Internally this metric runs dynamic time warping over per-slide lexical dissimilarity
    `(1 - token cosine similarity)` and then normalizes the final path cost into a 0..1 similarity score.
    The effect is to reward decks whose slide ordering and local topical progression can be aligned to the
    benchmark even when slide counts differ slightly.
    """
    if not generated_deck.slides or not reference_deck.slides:
        return 0.0, []
    rows = len(generated_deck.slides)
    cols = len(reference_deck.slides)
    cost = [[float("inf")] * (cols + 1) for _ in range(rows + 1)]
    backpointers: dict[tuple[int, int], tuple[int, int]] = {}
    cost[0][0] = 0.0

    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            similarity = token_cosine_similarity(
                generated_deck.slides[row - 1].combined_text,
                reference_deck.slides[col - 1].combined_text,
            )
            best_prev = min(
                (cost[row - 1][col], (row - 1, col)),
                (cost[row][col - 1], (row, col - 1)),
                (cost[row - 1][col - 1], (row - 1, col - 1)),
                key=lambda item: item[0],
            )
            cost[row][col] = (1.0 - similarity) + best_prev[0]
            backpointers[(row, col)] = best_prev[1]

    path: list[tuple[int, int]] = []
    row = rows
    col = cols
    while row > 0 and col > 0:
        path.append((row - 1, col - 1))
        row, col = backpointers[(row, col)]
    path.reverse()
    return normalize_dtw_cost(cost[rows][cols]), path


def deck_coherence_llm_metric(
    generated_deck: DeckView,
    llm_config_path: str,
    model_group: str = "evaluator",
) -> tuple[float, str]:
    """Use the DeckBench coherence prompt to judge full-deck logical flow and context.

    Internally this metric concatenates generated slide text, submits that summary to the DeckBench
    coherence rubric, and parses the returned JSON `{reason, score}` payload. The effect is to capture
    narrative quality, transitions, and contextual completeness that deterministic text metrics do not see.
    """
    init_from_config(llm_config_path)
    llm = get_llm(LLMConfig(model=model_group, temperature=0.0))
    response = llm.complete(
        [
            {"role": "system", "content": coherence_system_prompt},
            {"role": "user", "content": coherence_prompt.format(slides_gen="\n".join(slide.combined_text for slide in generated_deck.slides))},
        ],
        schema=DeckJudgeResponse,
    )
    payload = DeckJudgeResponse.model_validate_json(response)
    return float(payload.score), payload.reason

