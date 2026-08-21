from src.graph_state import GraphState

import re

def parse_and_validate_input(state: GraphState) -> GraphState:
    """Extracts patient_id from incoming metadata or parses it from user_query."""
    user_query = state.get("user_query", "")
    patient_id = state.get("patient_id")

    # Attempt to extract patient_id from user_query if not already present
    if not patient_id and user_query:
        # Regex to find patient IDs like P001, P123, etc.
        match = re.search(r"P\d{3}", user_query)
        if match:
            patient_id = match.group(0)
            print(f"Patient ID '{patient_id}' extracted from query.")

    missing_patient_id = patient_id is None

    print(f"--- Parsing and Validating Input ---")
    return {
        "patient_id": patient_id,
        "missing_patient_id": missing_patient_id,
        "audit_payload": {"input_parsed": True, "patient_id_found": not missing_patient_id}
    }

def request_patient_id(state: GraphState) -> GraphState:
    """Formats a response prompting the user to specify a patient when no patient_id is found."""
    print(f"--- Requesting Patient ID ---")
    return {
        "llm_output": "Please provide a patient identifier (e.g., P123) so I can retrieve their information.",
        "missing_patient_id": True,
        "audit_payload": {"patient_id_requested": True}
    }
