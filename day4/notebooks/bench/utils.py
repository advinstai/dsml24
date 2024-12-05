import os
import re
import numpy as np
import pandas as pd
from contextlib import contextmanager
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPEN_AI_ORGANIZATION = os.getenv("OPEN_AI_ORGANIZATION")


@contextmanager
def get_openai_client():
    try:
        client = OpenAI(
            organization=OPEN_AI_ORGANIZATION,
            api_key=OPENAI_API_KEY,
        )
        yield client
    except Exception as e:
        raise e
    finally:
        client.close()


def get_chat_completionprompt(model, messages) -> str:
    with get_openai_client() as client:
        response = client.chat.completions.create(
            model=model, messages=messages
        )
        return response.choices[0].message.content


def document_based_chunking(text, character_limit=500, lower_bound=150):
    paragraphs = re.split(r"[\n\r\t]", text)
    paragraphs = [p for p in paragraphs if len(p) > lower_bound]

    chunks = []

    for paragraph in paragraphs:
        sentences = re.split(r"\.|\?|\!", paragraph)
        punctuations = re.split(r"[^\.\?\!]", paragraph)
        punctuations = [p for p in punctuations if p]
        current_chunk = ""

        for index, sentence in enumerate(sentences):
            if index < len(punctuations):
                punctuation = punctuations[index]
            else:
                punctuation = "."
            if len(current_chunk) + len(sentence) <= character_limit:

                current_chunk += sentence + f"{punctuation} "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + f"{punctuation} "

        if current_chunk:
            chunks.append(current_chunk.strip())

    return chunks


def get_chunk_embeddings(text, model):
    with get_openai_client() as client:
        response = client.embeddings.create(model=model, input=text)
        return response.data[0].embedding


def vector_sim_retriever(
    database: pd.DataFrame,
    query: str,
    model: str = "text-embedding-3-small",
    top_k: int = 5,
):
    query_embedding = np.array(get_chunk_embeddings(query, model))
    database_embeddings = database["embeddings"].to_list()
    database_embeddings = np.array(database_embeddings)
    database_embeddings = database_embeddings.reshape(-1, len(query_embedding))
    nn = NearestNeighbors(n_neighbors=top_k, metric="cosine")
    nn.fit(database_embeddings)
    _, indices = nn.kneighbors(query_embedding.reshape(1, -1))
    return database.iloc[indices[0]]
