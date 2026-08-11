import os
import pandas as pd
import streamlit as st

from config import PDF_PATH, DATA_PATH
from rag.rag import ask_question
from vectorstore.vector_db import create_vectorstore


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG Benchmark System",
    page_icon="RAG",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = []

if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False


# ============================================================
# TITLE
# ============================================================

st.title("RAG Benchmark System")

st.write(
    "Compare embedding models and reasoning LLMs "
    "while answering questions from your uploaded PDF."
)


# ============================================================
# UPLOAD PDF
# ============================================================

st.subheader("1. Upload PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:

    os.makedirs(DATA_PATH, exist_ok=True)

    with open(PDF_PATH, "wb") as file:
        file.write(uploaded_file.getbuffer())

    st.success(
        f"PDF uploaded successfully: {uploaded_file.name}"
    )


# ============================================================
# EMBEDDING MODEL
# ============================================================

st.subheader("2. Select Embedding Model")

embedding_model = st.selectbox(
    "Embedding Model",
    [
        "minilm",
        "bge",
        "mpnet"
    ],
    format_func=lambda x: {
        "minilm": "MiniLM",
        "bge": "BGE",
        "mpnet": "MPNet"
    }[x]
)


# ============================================================
# REASONING LLM
# ============================================================

st.subheader("3. Select Reasoning LLM")

llm_model = st.selectbox(
    "Reasoning LLM",
    [
        "llama3.1",
        "gemma",
        "llama3.3"
    ],
    format_func=lambda x: {
        "llama3.1": "Llama 3.1",
        "gemma": "Gemma",
        "llama3.3": "Llama 3.3"
    }[x]
)


# ============================================================
# BUILD VECTOR DATABASE
# ============================================================

st.subheader("4. Build Vector Database")

if st.button(
    "Build Vector Database",
    type="primary"
):

    if uploaded_file is None:

        st.warning("Please upload a PDF first.")

    else:

        try:

            with st.spinner(
                f"Creating vector database using {embedding_model}..."
            ):

                vectordb = create_vectorstore(
                    embedding_model
                )

                document_count = (
                    vectordb._collection.count()
                )

            st.session_state.pdf_ready = True

            st.success(
                "Vector Database Created Successfully!"
            )

            st.info(
                f"Chunks stored in Chroma: {document_count}"
            )

        except Exception as e:

            st.error(
                f"Error while creating vector database: {e}"
            )


# ============================================================
# ASK QUESTION
# ============================================================

st.subheader("5. Ask a Question")

question = st.text_input(
    "Enter your question about the uploaded PDF",
    placeholder="Example: What are the technical skills?"
)


if st.button(
    "Get Answer",
    type="primary"
):

    if uploaded_file is None:

        st.warning("Please upload a PDF first.")

    elif not st.session_state.pdf_ready:

        st.warning(
            "Please build the vector database first."
        )

    elif question.strip() == "":

        st.warning("Please enter a question.")

    else:

        try:

            with st.spinner(
                "Searching PDF and generating answer..."
            ):

                answer, response_time, chunks = ask_question(
                    question,
                    embedding_model,
                    llm_model
                )

            # --------------------------------------------
            # ANSWER
            # --------------------------------------------

            st.subheader("Answer")

            st.write(answer)

            # --------------------------------------------
            # RETRIEVED CHUNKS
            # --------------------------------------------

            st.subheader("Retrieved Chunks")

            if chunks:

                for i, chunk in enumerate(
                    chunks,
                    start=1
                ):

                    with st.expander(
                        f"Chunk {i}"
                    ):

                        st.write(chunk)

            else:

                st.warning(
                    "No relevant chunks were retrieved."
                )

            # --------------------------------------------
            # RESPONSE TIME
            # --------------------------------------------

            st.subheader("Response Time")

            st.success(
                f"{response_time:.2f} seconds"
            )

            # --------------------------------------------
            # SAVE BENCHMARK RESULT
            # --------------------------------------------

            st.session_state.benchmark_results.append(
                {
                    "Embedding Model": embedding_model,
                    "Reasoning LLM": llm_model,
                    "Response Time (seconds)": response_time
                }
            )

        except Exception as e:

            st.error(
                f"Error while generating answer: {e}"
            )


# ============================================================
# BENCHMARK RESULTS
# ============================================================

st.divider()

st.subheader("Benchmark Results")


if st.session_state.benchmark_results:

    df = pd.DataFrame(
        st.session_state.benchmark_results
    )

    # Convert names for display
    df["Embedding Model"] = df[
        "Embedding Model"
    ].replace(
        {
            "minilm": "MiniLM",
            "bge": "BGE",
            "mpnet": "MPNet"
        }
    )

    df["Reasoning LLM"] = df[
        "Reasoning LLM"
    ].replace(
        {
            "llama3.1": "Llama 3.1",
            "gemma": "Gemma",
            "llama3.3": "Llama 3.3"
        }
    )

    # Benchmark table
    st.dataframe(
        df,
        hide_index=True
    )

    # --------------------------------------------
    # FASTEST COMBINATION
    # --------------------------------------------

    fastest_index = df[
        "Response Time (seconds)"
    ].idxmin()

    fastest = df.loc[
        fastest_index
    ]

    st.success(
        f"Fastest combination: "
        f"{fastest['Embedding Model']} + "
        f"{fastest['Reasoning LLM']} - "
        f"{fastest['Response Time (seconds)']:.2f} seconds"
    )

else:

    st.info(
        "No benchmark results yet. "
        "Ask a question to create a benchmark result."
    )