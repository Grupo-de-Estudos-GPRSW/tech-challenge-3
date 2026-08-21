from src.graph_state import GraphState

def guardrail_evaluator(state: GraphState) -> GraphState:
    """Inspects LLM output for sensitive keywords to determine if human approval is required."""
    llm_output = state["llm_output"]
    requires_human_approval = False

    print("--- Evaluating Guardrails ---")

    # Define a list of sensitive keywords that would trigger human approval
    sensitive_keywords = [
        "prescription", "prescribe",
        "administer", "administering",
        "perform surgery", "surgical procedure", "surgery",
        "diagnosis", "diagnose", # Often requires physician sign-off
        "change medication", "adjust dosage",
        "high-risk", "critical decision"
    ]

    # Check if any sensitive keywords are present in the LLM's output (case-insensitive)
    for keyword in sensitive_keywords:
        if keyword.lower() in llm_output.lower():
            requires_human_approval = True
            print(f"Guardrail triggered by keyword: '{keyword}'")
            break

    if requires_human_approval:
        print("Human approval REQUIRED due to sensitive content.")
    else:
        print("No sensitive content detected; human approval NOT required.")

    return {
        "requires_human_approval": requires_human_approval,
        "audit_payload": {**state.get("audit_payload", {}), "guardrail_evaluated": True, "requires_human_approval": requires_human_approval}
    }

def human_validation_gate(state: GraphState) -> GraphState:
    """Holds execution for physician sign-off on flagged high-risk actions."""
    print("--- Entering Human Validation Gate ---")
    # In a real system, this would trigger an alert, pause the workflow,
    # and wait for an external human approval step. For this simulation,
    # we will set the llm_output to a message indicating human approval is required.

    # Preserve the original LLM output in the audit payload for review
    original_llm_output = state["llm_output"]

    new_llm_output = (
        "The generated response contains sensitive content (e.g., medical advice, "
        "prescription recommendations). Human approval is required before this "
        "information can be delivered. The original suggestion has been sent for review."
    )

    print("Human approval required. Output marked for review.")

    return {
        "llm_output": new_llm_output,
        "audit_payload": {
            **state.get("audit_payload", {}),
            "human_validation_triggered": True,
            "original_llm_output_for_review": original_llm_output
        }
    }
