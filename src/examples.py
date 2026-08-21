from src.graph import app

print("\n--- Running Example 1: Query with Patient ID ---")
# Example 1: User query including a patient ID
initial_state_1 = {
    "user_query": "What is the recommended treatment for P001's diabetes?",
    "patient_id": None, # The parse_and_validate_input node will extract it
    "missing_patient_id": False,
    "patient_context": {},
    "retrieved_docs": [],
    "llm_output": "",
    "requires_human_approval": False,
    "audit_payload": {}
}

# Invoke the graph
output_state_1 = app.invoke(initial_state_1)
print("\nExample 1 Final Output (LLM Response):\n", output_state_1["llm_output"])
print("\nExample 1 Audit Payload (last entry):\n", output_state_1["audit_payload"])

print("\n--- Running Example 2: Query without Patient ID ---")
# Example 2: User query without a patient ID, expecting the system to ask for it
initial_state_2 = {
    "user_query": "What is the recommended protocol for managing anxiety?",
    "patient_id": None,
    "missing_patient_id": False,
    "patient_context": {},
    "retrieved_docs": [],
    "llm_output": "",
    "requires_human_approval": False,
    "audit_payload": {}
}

# Invoke the graph
output_state_2 = app.invoke(initial_state_2)
print("\nExample 2 Final Output (LLM Response):\n", output_state_2["llm_output"])
print("\nExample 2 Audit Payload (last entry):\n", output_state_2["audit_payload"])


print("\n--- Running Example 3: Query triggering Guardrail ---")
# Example 3: User query that might trigger guardrail (e.g., asking for a prescription)
initial_state_3 = {
    "user_query": "P003 has coronary artery disease. Should I prescribe them a new medication?",
    "patient_id": None,
    "missing_patient_id": False,
    "patient_context": {},
    "retrieved_docs": [],
    "llm_output": "",
    "requires_human_approval": False,
    "audit_payload": {}
}

# Invoke the graph
output_state_3 = app.invoke(initial_state_3)
print("\nExample 3 Final Output (LLM Response):\n", output_state_3["llm_output"])
print("\nExample 3 Audit Payload (last entry):\n", output_state_3["audit_payload"])
