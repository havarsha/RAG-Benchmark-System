from langchain_groq import ChatGroq


def get_llm(model_name):
    """
    Returns the selected Groq reasoning LLM.
    """

    if model_name == "llama3.1":

        return ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0
        )

    elif model_name == "gemma":

        return ChatGroq(
            model="gemma2-9b-it",
            temperature=0
        )

    elif model_name == "llama3.3":

        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0
        )

    else:
        raise ValueError(
            f"Unsupported LLM model: {model_name}"
        )