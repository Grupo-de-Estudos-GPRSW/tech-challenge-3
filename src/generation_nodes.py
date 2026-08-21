from src.graph_state import GraphState
from src.model_loading import model, tokenizer

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

def generate_llm_response(state: GraphState) -> GraphState:
    """Synthesizes context, protocols, and query using the fine-tuned LLM."""
    user_query = state["user_query"]
    patient_context = state["patient_context"]
    retrieved_docs = state.get("retrieved_docs", [])

    print(f"--- Generating LLM Response for: '{user_query}' ---")

    # Format patient context for the prompt
    patient_context_str = ""
    if patient_context:
        patient_context_str = "\nPatient's EHR Context:\n"
        for key, value in patient_context.items():
            if value: # Only add if value is not empty
                patient_context_str += f"- {key.replace('_', ' ').title()}: {', '.join(value) if isinstance(value, list) else value}\n"

    # Format retrieved protocols for the prompt
    retrieved_docs_str = ""
    if retrieved_docs:
        retrieved_docs_str = "\nRelevant Clinical Protocols:\n"
        for doc in retrieved_docs:
            retrieved_docs_str += f"- {doc}\n"

    # Define the prompt template based on the Alpaca format
    # We need to ensure the 'question' part of the Alpaca template encapsulates all context.
    alpaca_question_template = """###System:
You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, don't share false information.

### User:
{user_query_with_context}

### Assistant:
"""

    full_user_query_with_context = f"{patient_context_str}{retrieved_docs_str}{user_query}"

    prompt = PromptTemplate(
        template=alpaca_question_template,
        input_variables=["user_query_with_context"],
    )

    # Initialize HuggingFace pipeline with the pre-loaded model and tokenizer
    # Using text-generation pipeline
    text_generation_pipeline = pipeline(
        task="text-generation", # Explicitly specify the task
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512, # Adjust as needed
        pad_token_id=tokenizer.eos_token_id # Important for some models
    )
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    # Generate response
    # LangChain's .invoke() expects a dictionary with input_variables as keys
    chain = prompt | llm
    llm_response = chain.invoke({"user_query_with_context": full_user_query_with_context})

    print(f"--- Full user prompt with context: '{full_user_query_with_context}' ---")

    # The LLM's response will contain the prompt itself and the generated answer after '### Assistant:'
    # We need to extract only the generated answer.
    # Find the start of the assistant's response marker
    assistant_tag = "### Assistant:"
    if assistant_tag in llm_response:
        generated_text = llm_response.split(assistant_tag, 1)[1].strip()
    else:
        generated_text = llm_response.strip() # Fallback if tag is not found

    print(f"LLM Generated Response (first 100 chars): {generated_text[:100]}...")
    return {"llm_output": generated_text}
