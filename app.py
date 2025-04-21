# app.py
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import random

# streamlit run c:/Users/danie/GitHub/gf_is_hungry/app.py

# === UI SETUP ===
st.set_page_config(page_title="Hungry in Fremont 🍜", layout="centered")
st.title("🍽️ Fremont Restaurant Recommender")
st.markdown("Type a vibe or craving and we'll pick a place for you!")


# === LOAD EVERYTHING ===
@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    df = pd.read_pickle("restaurants.pkl")
    embeddings = np.load("embeddings.npy")
    index = faiss.read_index("faiss_index.index")
    return model, df, embeddings, index


model, df, _, index = load_resources()

query = st.text_input(
    "What are you in the mood for?",
    placeholder="e.g., spicy ramen, late night Korean, vegan brunch",
)

# === FILTERS ===
st.sidebar.header("🔍 Filters")
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 3.5, 0.5)
price_options = ["$", "$$", "$$$", "$$$$"]
max_price = st.sidebar.selectbox("Max Price Level", options=price_options, index=1)
category_input = st.sidebar.text_input(
    "Cuisine (optional)", placeholder="e.g., Korean, Indian, Sushi"
)

# === Session State for Try Again ===
if "filtered_results" not in st.session_state:
    st.session_state.filtered_results = pd.DataFrame()

# === ACTION BUTTONS ===
col1, col2 = st.columns([2, 1])
with col1:
    go_button = st.button("Surprise Me 🍀")
with col2:
    try_again_button = st.button("Try Again 🔁")


# === FILTERING LOGIC ===
def apply_filters(matched_df):
    def filter_row(row):
        try:
            rating_ok = float(row["rating"]) >= min_rating
            price_ok = row.get("price", "$$$$") <= max_price
            category_ok = (
                category_input.lower() in row["categories"].lower()
                if category_input
                else True
            )
            return rating_ok and price_ok and category_ok
        except:
            return False

    return matched_df[matched_df.apply(filter_row, axis=1)]


# === SEARCH FLOW ===
if go_button and query:
    query_vec = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, k=25)
    matched = df.iloc[I[0]]

    filtered = apply_filters(matched)
    st.session_state.filtered_results = filtered  # Save for retry

    if not filtered.empty:
        choice = filtered.sample(n=1).iloc[0]
        st.success(f"🎯 You should go to: {choice['name']}")
        st.write(f"**Categories:** {choice['categories']}")
        st.write(
            f"**Rating:** ⭐ {choice['rating']} | **Price:** 💵 {choice.get('price', 'N/A')}"
        )
        st.write(
            f"**Address:** {choice['address']}, {choice['city']}, {choice['state']}"
        )
        st.markdown(f"[🔗 Yelp Link]({choice['url']})")
    else:
        st.warning("No matching restaurants found with those filters.")

elif try_again_button and not st.session_state.filtered_results.empty:
    choice = st.session_state.filtered_results.sample(n=1).iloc[0]
    st.success(f"🎯 You should go to: {choice['name']}")
    st.write(f"**Categories:** {choice['categories']}")
    st.write(
        f"**Rating:** ⭐ {choice['rating']} | **Price:** 💵 {choice.get('price', 'N/A')}"
    )
    st.write(f"**Address:** {choice['address']}, {choice['city']}, {choice['state']}")
    st.markdown(f"[🔗 Yelp Link]({choice['url']})")
