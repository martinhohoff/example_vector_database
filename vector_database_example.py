import os
import sys
from typing import List

import chromadb
import pandas as pd
import tiktoken
from openai import OpenAI

# --- Config ---
EMBEDDING_MODEL = "text-embedding-3-small"
COST_PER_MILLION_TOKENS = 2.50  # Check OpenAI pricing for the specific model
CHROMA_DB_PATH = "./chroma_movies_db"
COLLECTION_NAME = "movies_openai_embeddings"
N_RESULTS = 3

# OpenAI client — set OPENAI_API_KEY in your environment variables
# https://platform.openai.com/account/api-keys
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def build_movie_text(row: pd.Series) -> str:
    return (
        f"Title: {row['title']} ({row['type']})\n"
        f"Description: {row['description']}\n"
        f"Categories: {row['listed_in']}"
    )


def count_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    return len(encoding.encode(text))


def get_embeddings(texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def print_results(results: dict) -> None:
    print("\nTop matches:\n")
    for i, (doc_id, document) in enumerate(zip(results["ids"][0], results["documents"][0])):
        print(f"Rank #{i + 1}")
        print("ID:", doc_id)
        print("Distance:", results["distances"][0]
              [i] if "distances" in results else None)
        print("Metadata:", results["metadatas"][0]
              [i] if "metadatas" in results else None)
        print("Document:")
        print(document)
        print("-" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python vector_database_example.py "<search query>"')
        sys.exit(1)

    query = sys.argv[1]

    df = pd.read_csv("movies_dataset.csv")
    print(df.columns)
    print(df.shape)

    df["text"] = df.apply(build_movie_text, axis=1)
    movie_texts = df["text"].tolist()

    total_tokens = sum(count_tokens(t) for t in movie_texts)
    total_cost = (total_tokens / 1_000_000) * COST_PER_MILLION_TOKENS
    print(f"Total tokens: {total_tokens}")
    print(f"Estimated embedding cost: ${total_cost:.4f}")

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    collection.upsert(
        ids=df["id"].tolist(),
        documents=movie_texts,
        embeddings=get_embeddings(movie_texts),
        metadatas=[
            {"title": row["title"], "type": row["type"],
                "listed_in": row["listed_in"]}
            for _, row in df.iterrows()
        ],
    )
    print("Inserted movies into Chroma.")

    query_embedding = get_embeddings([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding], n_results=N_RESULTS)
    print_results(results)
