import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
import os
load_dotenv()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-3-large"
            )

chroma_client = chromadb.PersistentClient(path="chroma_db")

# switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
collection = chroma_client.get_or_create_collection(name="data_test")

# switch `add` to `upsert` to avoid adding the same documents every time
collection.upsert(
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ],
    ids=["id1", "id2"]
)

results = collection.query(
    query_texts=["This apple"], # Chroma will embed this for you
    n_results=1 # how many results to return
)

print(results)
