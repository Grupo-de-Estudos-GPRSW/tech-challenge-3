from typing import TypedDict, List, Dict, Any

class GraphState(TypedDict):
    """Represent a state for our graph."""
    patient_id: str
    missing_patient_id: bool
    user_query: str
    patient_context: Dict[str, Any]  # Structured record of patient data
    retrieved_docs: List[str]  # Document chunks and citations
    llm_output: str  # Draft response from LLM
    requires_human_approval: bool  # Flag for safety evaluation
    audit_payload: Dict[str, Any]  # Log metadata
