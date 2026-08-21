from src.graph_state import GraphState
from src.patient_database import mock_patient_db

def fetch_ehr_context(state: GraphState) -> GraphState:
    """Queries structured patient databases to retrieve history and check pending tests."""
    patient_id = state["patient_id"]
    patient_context = {}
    for patient in mock_patient_db:
        if patient["patient_id"] == patient_id:
            patient_context = {
                "medical_history": patient["medical_history"],
                "current_medications": patient["current_medications"],
                "pending_tests": patient["pending_tests"]
            }
            break

    if not patient_context:
        print(f"Warning: Patient {patient_id} not found in mock database.")

    print(f"--- Fetched EHR Context for Patient {patient_id} ---")
    return {"patient_context": patient_context}
