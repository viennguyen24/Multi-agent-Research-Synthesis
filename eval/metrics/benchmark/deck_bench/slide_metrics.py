from __future__ import annotations

from pydantic import BaseModel

from eval.metrics.benchmark.deck_bench.math_utils import (
    EmbedderProtocol,
    embedding_cosine_similarity,
    token_cosine_similarity,
)
from eval.metrics.benchmark.deck_bench.prompts import content_prompt, content_system_prompt
from eval.schema import DeckViewSlide
from src.llm.llm import LLMConfig, get_llm, init_from_config


class SlideJudgeResponse(BaseModel):
    reason: str
    score: int


def slide_text_similarity_metric(generated_slide: DeckViewSlide, reference_slide: DeckViewSlide) -> float:
    """Measure lexical overlap between generated and reference slide text.

    Internally this metric builds bag-of-words counters for each slide and computes cosine similarity
    over token frequencies. The effect is to reward slides that preserve the same topical vocabulary
    and phrasing patterns as the benchmark reference without requiring exact string equality.
    """
    return token_cosine_similarity(generated_slide.combined_text, reference_slide.combined_text)


def slide_faithfulness_metric(generated_slide: DeckViewSlide, source_chunks: list[str]) -> float:
    """Measure whether one generated slide stays anchored to at least one source chunk.

    Internally this metric compares the generated slide text against every available source chunk with
    lexical cosine similarity and returns the best match. The effect is to detect whether the slide has
    a strong grounding anchor in the source material instead of drifting into unsupported claims.
    """
    if not source_chunks:
        return 0.0
    return max(token_cosine_similarity(generated_slide.combined_text, chunk) for chunk in source_chunks)


def slide_semantic_similarity_metric(
    generated_slide: DeckViewSlide,
    reference_slide: DeckViewSlide,
    embedder: EmbedderProtocol,
) -> float:
    """Measure semantic agreement between generated and reference slide text embeddings.

    Internally this metric embeds both slides with the shared sentence-transformer model and scores
    cosine similarity in embedding space. The effect is to reward slides that convey the same meaning
    as the reference even when wording changes enough to reduce token-level overlap.
    """
    return embedding_cosine_similarity(
        generated_slide.combined_text,
        reference_slide.combined_text,
        embedder,
    )


def slide_content_quality_llm_metric(
    generated_slide: DeckViewSlide,
    llm_config_path: str,
    model_group: str = "evaluator",
) -> tuple[float, str]:
    """Use the DeckBench slide-content prompt to judge clarity and text-image complementarity.

    Internally this metric sends the generated slide text through the DeckBench content rubric and
    parses the returned JSON `{reason, score}` payload. The effect is to capture presentation quality
    traits that deterministic text matching cannot represent, such as narrative clarity and perceived
    completeness.
    """
    init_from_config(llm_config_path)
    llm = get_llm(LLMConfig(model=model_group, temperature=0.0))
    response = llm.complete(
        [
            {"role": "system", "content": content_system_prompt},
            {"role": "user", "content": content_prompt.format(slides_gen=generated_slide.combined_text)},
        ],
        schema=SlideJudgeResponse,
    )
    payload = SlideJudgeResponse.model_validate_json(response)
    return float(payload.score), payload.reason

