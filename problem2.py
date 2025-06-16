import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# Load CSV
df = pd.read_csv("data/ex_2_data.csv")

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create Chroma client & collection
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="text_search")

# Prepare data
texts = df["paragraph"].tolist()
ids = df["id"].astype(str).tolist()
metadatas = df[["source", "category"]].to_dict(orient="records")

# Generate and add embeddings to Chroma
embeddings = model.encode(texts, convert_to_numpy=True).tolist()
collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)

# Define search function
def semantic_search(query: str, top_k=5):
    query_embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results

query = "renewable energy advantages"
results = semantic_search(query)

for i, (text, metadata, dist) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0])):
    print(f"Rank {i+1} | Distance: {dist:.4f}")
    print(f"Text: {text}")
    print(f"Metadata: {metadata}\n")


# generate embeddings in batches
# use more powerful vector db
# hybrid search (1 - filter texts using keywords (elastic), 2 - reranking using semantic search)

