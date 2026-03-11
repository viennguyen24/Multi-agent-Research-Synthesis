import operator
from typing import Annotated, Any, Literal, TypedDict


class ResearchState(TypedDict):
    query: str
    source_chunks: list[dict[str, Any]]
    selected_chunk_indices: list[int]
    manifest_json: dict[str, Any]
    plan: str
    draft: str
    critique: str
    iteration: int
    next: Literal["continue", "done"]
    messages: Annotated[list[str], operator.add]
