# file: query_rag.py
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERSIST_DIR = "./chroma-store"

# 1. Re‑load vector store
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=OPENAI_API_KEY,
)

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 2. LLM
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.2,
    api_key=OPENAI_API_KEY,
)

# 3. Build RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff",
)

def ask(query: str):
    result = qa_chain.invoke({"query": query})
    print("\nQ:", query)
    print("\nA:", result["result"])
    print("\nSources:")
    for i, d in enumerate(result["source_documents"], start=1):
        print(f"- [{i}] {d.page_content[:120]}...")

if __name__ == "__main__":
    ask("Explain RAG to a junior backend developer.")
