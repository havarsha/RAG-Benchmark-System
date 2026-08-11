import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import PDF_PATH, DB_PATH
from embeddings.embedding import get_embedding


def get_collection_name(embedding_model):
    return f"rag_{embedding_model}"


def create_vectorstore(embedding_model):

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    # Load PDF
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    if not documents:
        raise ValueError(
            "No text could be extracted from the PDF."
        )

    # Split PDF
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "No chunks were created from the PDF."
        )

    # Embedding
    embedding = get_embedding(embedding_model)

    # Make sure DB directory exists
    os.makedirs(DB_PATH, exist_ok=True)

    collection_name = get_collection_name(
        embedding_model
    )

    # Create Chroma database
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=DB_PATH
    )

    # Add documents
    vectordb.add_documents(chunks)

    return vectordb


def load_vectorstore(embedding_model):

    embedding = get_embedding(embedding_model)

    collection_name = get_collection_name(
        embedding_model
    )

    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=DB_PATH
    )

    return vectordb