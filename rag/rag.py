import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from llms.llm import get_llm
from vectorstore.vector_db import load_vectorstore


def ask_question(question, embedding_model, llm_model):
    """
    Retrieve relevant chunks from the PDF and
    generate an answer using the selected Groq LLM.
    """

    # -----------------------------------
    # Load vector database
    # -----------------------------------

    vectordb = load_vectorstore(embedding_model)

    # -----------------------------------
    # Check whether documents exist
    # -----------------------------------

    collection = vectordb._collection

    document_count = collection.count()

    if document_count == 0:
        return (
            "No documents found in the vector database. "
            "Please build the vector database first.",
            0.0,
            []
        )

    # -----------------------------------
    # Retriever
    # -----------------------------------

    retriever = vectordb.as_retriever(
        search_kwargs={"k": 3}
    )

    # -----------------------------------
    # LLM
    # -----------------------------------

    llm = get_llm(llm_model)

    # -----------------------------------
    # Prompt
    # -----------------------------------

    prompt = ChatPromptTemplate.from_template(
        """
You are a document question-answering assistant.

Your job is to answer the user's question using ONLY
the information contained in the provided context.

Do NOT use outside knowledge.

If the answer cannot be found in the context,
respond exactly:

I don't know based on the provided document.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    # -----------------------------------
    # Format retrieved documents
    # -----------------------------------

    def format_documents(documents):
        return "\n\n".join(
            document.page_content
            for document in documents
        )

    # -----------------------------------
    # RAG chain
    # -----------------------------------

    rag_chain = (
        {
            "context": retriever | format_documents,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # -----------------------------------
    # Measure response time
    # -----------------------------------

    start_time = time.perf_counter()

    answer = rag_chain.invoke(question)

    end_time = time.perf_counter()

    response_time = round(
        end_time - start_time,
        2
    )

    # -----------------------------------
    # Retrieve chunks separately
    # -----------------------------------

    retrieved_documents = retriever.invoke(question)

    retrieved_chunks = [
        document.page_content
        for document in retrieved_documents
    ]

    return (
        answer,
        response_time,
        retrieved_chunks
    )