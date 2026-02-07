# file: query_rag.py
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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

# 3. Build retrieval chain
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know.\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

def format_documents(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {"context": retriever | format_documents, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def ask(query: str):
    answer = qa_chain.invoke(query)
    print("\nQ:", query)
    print("\nA:", answer)
    # Get sources separately for display
    docs = retriever.invoke(query)
    print("\nSources:")
    for i, d in enumerate(docs, start=1):
        print(f"- [{i}] {d.page_content[:120]}...")

if __name__ == "__main__":
    ask("Explain RAG to a junior backend developer.")
