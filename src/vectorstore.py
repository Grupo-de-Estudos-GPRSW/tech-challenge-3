from src.protocols_database import mock_protocols

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

protocols_doc = [Document(page_content=protocol["content"], metadata={"protocol_id": protocol["protocol_id"], "protocol_title": protocol["title"] }) for protocol in mock_protocols]

vectorstore = FAISS.from_documents(protocols_doc, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
