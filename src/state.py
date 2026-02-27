import operator
from typing import Annotated, Literal, TypedDict


class ResearchState(TypedDict):
    query: str
    plan: str
    draft: str
    critique: str
    iteration: int
    next: Literal["continue", "done"]
    messages: Annotated[list[str], operator.add]
