# file: build_index.py
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 1. Your corpus (replace with file loaders in your training later)
raw_docs = [
    "Agentic AI systems can autonomously decide which tools to call.",
    "RAG combines retrieval from a vector store with generation from an LLM.",
    "Chroma is an open‑source vector database optimized for AI applications.",
]

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
)
docs = text_splitter.split_documents(
    [Document(page_content=t) for t in raw_docs]
)

# 3. Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=OPENAI_API_KEY,
)

# 4. Create / persist Chroma store
PERSIST_DIR = "./chroma-store"

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=PERSIST_DIR,
)

print("Chroma index built at", PERSIST_DIR)
