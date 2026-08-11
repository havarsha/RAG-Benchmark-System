from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding(model_name):
    """
    Returns the selected local embedding model.
    No API key is required.
    """

    if model_name == "minilm":

        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    elif model_name == "bge":

        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

    elif model_name == "mpnet":

        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

    else:
        raise ValueError(
            f"Unsupported embedding model: {model_name}"
        )