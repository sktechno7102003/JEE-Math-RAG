import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

# --- CONFIG: keep in sync with ingestion_pipeline.py ---
CHROMA_DIR = os.path.join("chroma_db", "jee_math_pyq")
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
# Groq model to use for answering questions
GROQ_MODEL_NAME = "qwen/qwen3-32b"


# ... (you can keep QAChunk / other dataclasses if you still use them elsewhere) ...

encode_kwargs = {
        'normalize_embeddings': True, 
        'batch_size': 4 
    }
def get_embeddings() -> HuggingFaceEmbeddings:
    """Create HuggingFace embeddings via LangChain."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,
                                 encode_kwargs=encode_kwargs)


def load_vectorstore() -> Chroma:
    """
    Load an existing Chroma DB that was populated by ingestion_pipeline.py.

    This function does NOT ingest or modify any embeddings.
    It only loads what is already there.
    """
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        raise FileNotFoundError(
            f"Chroma DB at '{CHROMA_DIR}' not found or empty.\n"
            "Run the ingestion pipeline first, e.g.:\n"
            "  python -m src.ingestion_pipeline"
        )

    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )


def get_llm() -> ChatGroq:
    """Create a Groq chat model via LangChain."""
    return ChatGroq(
        model=GROQ_MODEL_NAME,
        temperature=0,      # deterministic, good for factual RAG
        max_tokens=None,    # no hard cap; Groq will enforce model limits
        timeout=None,
        max_retries=2,
    )


def build_rag_chain(vectorstore: Chroma):
    """Build a LangChain RAG pipeline: retriever -> prompt -> Groq."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = get_llm()

    system_template = (
        "You are an expert JEE Mathematics tutor.\n"
        "You are given previous year questions with solutions as context.\n"
        "Use ONLY that context to answer the student's query.\n"
        "If the context is not enough, say you are not sure instead of guessing.\n"
        "When helpful, reference the question IDs from the context.\n\n"
        "Context:\n{context}\n\n"
        "Student's question:\n{question}\n"
        "Explain step by step in a way that a JEE aspirant can follow."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
        ]
    )

    setup = RunnableParallel(
        context=retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        question=RunnablePassthrough(),
    )

    rag_chain = setup | prompt | llm
    return rag_chain


def main() -> None:
    # --- NEW: just load the existing vector store, do NOT ingest ---
    print("Loading existing Chroma vector store...")
    vectorstore = load_vectorstore()

    print("Building RAG chain with Groq + HuggingFace embeddings + Chroma...")
    rag_chain = build_rag_chain(vectorstore)

    print(
        "\nRAG JEE Math assistant is ready (LangChain + Chroma).\n"
        "Type your doubt (or 'exit' to quit).\n"
    )

    while True:
        try:
            user_q = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_q:
            continue
        if user_q.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            response = rag_chain.invoke(user_q)
        except Exception as e:
            print(f"Error while running RAG chain: {e}")
            continue

        print("\nAssistant:\n")
        print(getattr(response, "content", str(response)))


if __name__ == "__main__":
    main()