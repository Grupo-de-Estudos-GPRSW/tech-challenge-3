from src.protocols_database import mock_protocols

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

OPENAI_API_KEY = "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"

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

protocols_doc = [Document(page_content=protocol["content"], metadata={"protocol_id": protocol["protocol_id"], "protocol_title": protocol["title"] }) for protocol in mock_protocols]

vectorstore = FAISS.from_documents(protocols_doc, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
