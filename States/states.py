from typing_extensions import TypedDict
from typing import Dict, List, Any

class SharedState(TypedDict, total=False):
    question: str
    classification: str
    search_results: Dict[str, str]
    verified_claims: str
    report: str
    quality_feedback: str
    messages: List[Dict[str, Any]]
    remaining_steps: int