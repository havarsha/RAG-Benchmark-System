import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import PDF_PATH, DB_PATH
from embeddings.embedding import get_embedding


def get_collection_name(
    embedding_model,
    chunk_size,
    chunk_overlap
):
    """
    Creates a unique Chroma collection name
    for each chunking configuration.
    """

    return (
        f"rag_{embedding_model}_"
        f"{chunk_size}_{chunk_overlap}"
    )


def create_vectorstore(
    embedding_model,
    chunk_size=800,
    chunk_overlap=150
):
    """
    Loads the PDF, splits it into chunks using
    the selected chunk size and overlap, and
    stores the chunks in Chroma.
    """

    # Check PDF
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    # ---------------------------------------------
    # Load PDF
    # ---------------------------------------------

    loader = PyPDFLoader(PDF_PATH)

    documents = loader.load()

    if not documents:
        raise ValueError(
            "No content could be extracted from the PDF."
        )

    # ---------------------------------------------
    # Chunking
    # ---------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "No chunks were created from the PDF."
        )

    # ---------------------------------------------
    # Embedding
    # ---------------------------------------------

    embedding = get_embedding(
        embedding_model
    )

    # ---------------------------------------------
    # Chroma
    # ---------------------------------------------

    os.makedirs(
        DB_PATH,
        exist_ok=True
    )

    collection_name = get_collection_name(
        embedding_model,
        chunk_size,
        chunk_overlap
    )

    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=DB_PATH
    )

    # Add chunks
    vectordb.add_documents(chunks)

    return vectordb


def load_vectorstore(
    embedding_model,
    chunk_size=800,
    chunk_overlap=150
):
    """
    Loads the Chroma database corresponding
    to the selected embedding and chunking setup.
    """

    embedding = get_embedding(
        embedding_model
    )

    collection_name = get_collection_name(
        embedding_model,
        chunk_size,
        chunk_overlap
    )

    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=DB_PATH
    )

    return vectordb