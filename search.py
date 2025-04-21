import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import random

TOP_K = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# -------------------------------------------------------------------------------------
"""Load embeddings"""
df = pd.read_pickle("restaurants.pkl")
embeddings = np.load("embeddings.npy")
index = faiss.read_index("faiss_index.index")
model = SentenceTransformer(EMBEDDING_MODEL)

# Search loop
print("\n🔍 Ready! Type your food vibe (e.g., 'cozy ramen late night'):")
print("Type 'exit' to quit.\n")

while True:
    query = input("🔎 Your query: ").strip()
    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting.")
        break

    query_vec = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, k=TOP_K)

    matched_rows = df.iloc[I[0]]
    print("\n✨ Top matches:")
    for idx, row in matched_rows.iterrows():
        print(
            f"- {row['name']} | {row['categories']} | {row['address']} | ⭐ {row['rating']} | 💵 {row.get('price', 'N/A')}"
        )

    selected = matched_rows.sample(n=1).iloc[0]
    print("\n🎯 You should go to:")
    print(f"👉 {selected['name']} | {selected['categories']}")
    print(f"📍 {selected['address']}, {selected['city']}, {selected['state']}")
    print(f"⭐ {selected['rating']} stars | 💵 {selected.get('price', 'N/A')}")
    print(f"🔗 {selected['url']}")
    print("\n" + "-" * 50 + "\n")
