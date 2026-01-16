import os

# Avoid importing TensorFlow/Keras inside Transformers so we don't hit the
# "Keras 3 is not supported" error. We only need the PyTorch stack.
os.environ["TRANSFORMERS_NO_TF"] = "1"

from dotenv import load_dotenv
import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


load_dotenv()

# --- CONFIG: keep in sync with 1_ingestion_pipeline.py and 2_answer_generation.py ---
CHROMA_DIR = os.path.join("chroma_db", "jee_math_pyq")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Groq model to use for answering questions
GROQ_MODEL_NAME = "qwen/qwen3-32b"


@st.cache_resource(show_spinner=False)
def get_vectorstore() -> Chroma:
    """Load an existing Chroma DB created by the ingestion pipeline."""
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        raise FileNotFoundError(
            f"Chroma DB at '{CHROMA_DIR}' not found or empty. "
            "Run the ingestion pipeline first (1_ingestion_pipeline.py)."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatGroq:
    """Create a cached Groq chat model."""
    return ChatGroq(
        model=GROQ_MODEL_NAME,
        temperature=0,      # deterministic, good for factual RAG
        max_tokens=None,    # no hard cap; Groq will enforce model limits
        timeout=None,
        max_retries=2,
    )


@st.cache_resource(show_spinner=False)
def get_rag_chain():
    """Build and cache the RAG chain and retriever."""
    vectorstore = get_vectorstore()
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
        context=retriever
        | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        question=RunnablePassthrough(),
    )

    rag_chain = setup | prompt | llm
    return rag_chain, retriever


def main() -> None:
    st.set_page_config(page_title="JEE Math RAG Tester", page_icon="🧮", layout="wide")
    st.title("🧮 JEE Mathematics RAG Pipeline Tester")

    st.markdown(
        """
This app lets you **test your ingestion + retrieval pipeline** on JEE Math PYQs.

- The questions and solutions are stored as **LaTeX** in your JSON and Chroma DB.
- Streamlit renders LaTeX using standard `$...$` / `$$...$$` syntax.
- The model uses **only the retrieved context** from your vector store.
        """
    )

    with st.sidebar:
        st.header("Pipeline status")
        try:
            _ = get_vectorstore()
            st.success("Chroma DB loaded ✅")
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Error loading vector store: {e}")
            st.stop()

        st.info(
            "If you add new papers to `data/`, run `1_ingestion_pipeline.py` again "
            "to update the embeddings. This app will automatically see the changes."
        )
    # --- Simple chat-style interface ---
    st.subheader("Chat with your JEE Math assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input(
        "Ask a JEE Mathematics doubt (supports LaTeX, e.g. `$\\int_0^1 x^2 dx$`)"
    )

    if user_q:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    rag_chain, _retriever = get_rag_chain()

                    # RAG chain internally:
                    #   1. Uses the retriever to get relevant chunks from Chroma
                    #   2. Builds the prompt with {context} and {question}
                    #   3. Calls the Groq LLM to generate the answer
                    response = rag_chain.invoke(user_q.strip())
                    answer_text = getattr(response, "content", str(response))
                except Exception as e:
                    st.error(f"Error while running RAG chain: {e}")
                    return

                # Show model answer (LaTeX will render)
                st.markdown(answer_text)

        st.session_state.messages.append({"role": "assistant", "content": answer_text})


if __name__ == "__main__":
    main()

