from dataclasses import dataclass
from typing import List, Dict, TypedDict, Annotated
import operator

@dataclass
class Agent:
    """Agent class for different LLM agents."""
    name: str
    instructions: str

class GraphState(TypedDict):
    """Graph state for information propagation."""
    question: str
    generation: str
    web_search: str
    max_retries: int
    answers: int
    loop_step: Annotated[int, operator.add]
    documents: List[str]