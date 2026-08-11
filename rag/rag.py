import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

from llms.llm import get_llm
from vectorstore.vector_db import load_vectorstore


def ask_question(
    question,
    embedding_model,
    llm_model,
    chunk_size=800,
    chunk_overlap=150
):
    """
    Answers a question using the uploaded PDF.

    The vector database is selected according to:
    - embedding model
    - chunk size
    - chunk overlap

    Returns:
        answer
        response_time
        retrieved_chunks
    """

    # --------------------------------------------------
    # Load the correct Vector Database
    # --------------------------------------------------

    vectordb = load_vectorstore(
        embedding_model,
        chunk_size,
        chunk_overlap
    )

    # --------------------------------------------------
    # Retriever
    # --------------------------------------------------

    retriever = vectordb.as_retriever(
        search_kwargs={
            "k": 3
        }
    )

    # --------------------------------------------------
    # LLM
    # --------------------------------------------------

    llm = get_llm(
        llm_model
    )

    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful AI assistant.

Answer the user's question using ONLY
the information provided in the context.

Do not make up information.

If the answer cannot be found in the
provided context, reply exactly:

I don't know based on the provided document.

Context:
{context}
"""
            ),
            (
                "human",
                "{input}"
            )
        ]
    )

    # --------------------------------------------------
    # Document Chain
    # --------------------------------------------------

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    # --------------------------------------------------
    # Retrieval Chain
    # --------------------------------------------------

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    # --------------------------------------------------
    # Measure Response Time
    # --------------------------------------------------

    start_time = time.perf_counter()

    response = retrieval_chain.invoke(
        {
            "input": question
        }
    )

    end_time = time.perf_counter()

    response_time = round(
        end_time - start_time,
        2
    )

    # --------------------------------------------------
    # Retrieved Chunks
    # --------------------------------------------------

    retrieved_chunks = []

    for doc in response.get(
        "context",
        []
    ):
        retrieved_chunks.append(
            doc.page_content
        )

    # --------------------------------------------------
    # Return Results
    # --------------------------------------------------

    return (
        response["answer"],
        response_time,
        retrieved_chunks
    )