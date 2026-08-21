from src.graph_state import GraphState
from src.input_nodes import parse_and_validate_input, request_patient_id
from src.ehr_nodes import fetch_ehr_context
from src.retrieval_nodes import retrieve_protocols
from src.generation_nodes import generate_llm_response
from src.guardrail_nodes import guardrail_evaluator, human_validation_gate
from src.audit_nodes import audit_logger

from langgraph.graph import StateGraph, END

# Define the LangGraph workflow
workflow = StateGraph(GraphState)

# Add nodes to the graph
workflow.add_node("parse_and_validate_input", parse_and_validate_input)
workflow.add_node("request_patient_id", request_patient_id)
workflow.add_node("fetch_ehr_context", fetch_ehr_context)
workflow.add_node("retrieve_protocols", retrieve_protocols)
workflow.add_node("generate_llm_response", generate_llm_response)
workflow.add_node("guardrail_evaluator", guardrail_evaluator)
workflow.add_node("human_validation_gate", human_validation_gate)
workflow.add_node("audit_logger", audit_logger)

# Set the entry point of the graph
workflow.set_entry_point("parse_and_validate_input")

# Define conditional edge from parse_and_validate_input
def route_input(state: GraphState):
    if state["missing_patient_id"]:
        return "request_patient_id"
    else:
        return "fetch_ehr_context"

workflow.add_conditional_edges(
    "parse_and_validate_input", # Source node
    route_input,                # Conditional function
    {
        "request_patient_id": "request_patient_id",
        "fetch_ehr_context": "fetch_ehr_context"
    }
)

# Add direct edges
workflow.add_edge("request_patient_id", "audit_logger") # After requesting patient ID, log and end
workflow.add_edge("fetch_ehr_context", "retrieve_protocols")
workflow.add_edge("retrieve_protocols", "generate_llm_response")
workflow.add_edge("generate_llm_response", "guardrail_evaluator")

# Define conditional edge from guardrail_evaluator
def route_guardrail(state: GraphState):
    if state["requires_human_approval"]:
        return "human_validation_gate"
    else:
        return "audit_logger"

workflow.add_conditional_edges(
    "guardrail_evaluator", # Source node
    route_guardrail,       # Conditional function
    {
        "human_validation_gate": "human_validation_gate",
        "audit_logger": "audit_logger"
    }
)

workflow.add_edge("human_validation_gate", "audit_logger") # After human validation, log and end

# Finally, define the END point for audit_logger
workflow.add_edge("audit_logger", END)

# Compile the graph
app = workflow.compile()

print("LangGraph nodes and edges defined, and graph compiled.")
