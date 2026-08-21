from src.graph_state import GraphState
from src.vectorstore import retriever

def retrieve_protocols(state: GraphState) -> GraphState:
    query = state.get("user_query", "")
    # O método correto nas versões atuais é .invoke()
    hits = retriever.invoke(query)
    # Guardamos apenas texto e metadados necessários

    retrieved_docs = []
    [retrieved_docs.append(f"Protocol ID: {d.metadata.get("protocol_id", "")}, Title: {d.metadata.get("protocol_title", "")}\nContent: {d.page_content}") for d in hits]

    return {"retrieved_docs": retrieved_docs}
