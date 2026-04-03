# Example Vector Database

This repository shows a minimal vector search workflow for movie data using:

- OpenAI embeddings
- ChromaDB as the local vector store
- A CSV dataset of movies

The script reads movie records from `movies_dataset.csv`, generates embeddings for each movie description, stores them in a persistent Chroma collection, and runs a semantic search query against that collection.

## Files

- `vector_database_example.py` - main example script
- `movies_dataset.csv` - input dataset used to build the collection

## What The Script Does

1. Loads the movie dataset with pandas
2. Builds a searchable text field from title, type, description, and categories
3. Counts tokens to estimate embedding cost
4. Requests embeddings from the OpenAI API
5. Stores documents, embeddings, and metadata in a local Chroma database
6. Queries the collection using a natural-language search string
7. Prints the top matching results

## Requirements

- Python 3.9+
- An OpenAI API key

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

## Run

From the project root:

```bash
python vector_database_example.py
```

## Configuration

The main settings are defined near the top of `vector_database_example.py`:

- `EMBEDDING_MODEL` - embedding model used for vector creation
- `COST_PER_MILLION_TOKENS` - estimated embedding cost reference
- `CHROMA_DB_PATH` - local folder where Chroma stores data
- `COLLECTION_NAME` - collection name inside Chroma
- `QUERY` - semantic search prompt
- `N_RESULTS` - number of matches returned

## Output

When the script runs, it prints:

- dataset columns and shape
- total token count
- estimated embedding cost
- confirmation that movies were inserted into Chroma
- the top semantic search matches, including distance, metadata, and document text

## Notes

- The Chroma database is persisted locally in `./chroma_movies_db`.
- Running the script repeatedly will reuse the same collection name and upsert the same IDs.
- The embedding cost printed by the script is only an estimate based on the configured rate.
- The script expects `movies_dataset.csv` to contain fields such as `id`, `title`, `type`, `description`, and `listed_in`.

## Example Query

The current default query is:

```text
A mind-bending science fiction movie about space or dreams
```

You can edit `QUERY` in `vector_database_example.py` to test different semantic searches.
