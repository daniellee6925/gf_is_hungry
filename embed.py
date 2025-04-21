import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# -------------------------------------------------------------------------------------
"""CONFIG"""
CSV_FILE = "fremont_restaurants.csv"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5  # Number of search results to return

# -------------------------------------------------------------------------------------
"""Load Data"""
df = pd.read_csv(CSV_FILE)

# -------------------------------------------------------------------------------------
"""Create Text for Embedding"""
texts = (
    df["name"]
    + " - "
    + df["categories"]
    + " - "
    + df["address"].fillna("")
    + " - "
    + df["city"]
    + ", "
    + df["state"]
    + " - Rating: "
    + df["rating"].astype(str)
    + " - Price: "
    + df["price"].fillna("")
).tolist()

# -------------------------------------------------------------------------------------
"""Embed text using embedding model"""
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

# Save embeddings and dataframe
np.save("embeddings.npy", embeddings)
df.to_pickle("restaurants.pkl")
# -------------------------------------------------------------------------------------
"""Build FAISS Index"""
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "faiss_index.index")

print("Embedding and indexing complete. Files saved:")
print("  - faiss_index.index")
print("  - embeddings.npy")
print("  - restaurants.pkl")
