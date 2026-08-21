from src.graph_state import GraphState

import json # For pretty printing the audit payload
import os
import datetime

def audit_logger(state: GraphState) -> GraphState:
    """Logs the final execution trace and validation states to audit storage."""
    print("--- Logging Audit Trail ---")
    final_llm_output = state.get("llm_output", "No LLM output generated.")
    audit_payload = state.get("audit_payload", {})

    # Define log directory and ensure it exists
    log_directory = "execution_log"
    os.makedirs(log_directory, exist_ok=True)

    # Generate a unique filename using a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_filename = f"audit_log_{timestamp}.txt"
    log_filepath = os.path.join(log_directory, log_filename)

    # Write to a persistent log file
    with open(log_filepath, "w") as f:
        f.write("Final LLM Output:\n")
        f.write(final_llm_output)
        f.write("\n\nAudit Payload:\n")
        f.write(json.dumps(audit_payload, indent=2))

    print(f"Audit trail successfully logged to: {log_filepath}")

    # Update the audit payload to confirm logging happened and record the path
    return {
        "audit_payload": {**audit_payload, "logged_at": str(datetime.datetime.now()), "log_filepath": log_filepath}
    }
