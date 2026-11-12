
import pandas as pd
import requests
import streamlit as st
import time
from pathlib import Path

CSV_PATH = Path("dataset/combined_movies.csv")

API_KEY = st.secrets["TMDB_API_KEY"]
BASE_URL = "https://api.themoviedb.org/3/search/movie"
IMAGE_URL = "https://image.tmdb.org/t/p/w500"

df = pd.read_csv(CSV_PATH)
df["poster_path"] = df["poster_path"].fillna("").astype(str)

# Only process first 100 Hollywood movies without posters
mask = (
    (df["category"] == "Hollywood") &
    (df["poster_path"].str.strip() == "")
)

movie_indices = df[mask].head(300).index

print("Movies selected for testing:", len(movie_indices))

session = requests.Session()
session.headers.update({"User-Agent": "CineMatch/1.0"})

success = 0
failed = 0

for count, index in enumerate(movie_indices, start=1):

    title = str(df.at[index, "title"]).strip()
    print(f"\n[{count}/300] Searching: {title}")

    params = {"api_key": API_KEY, "query": title}

    try:
        response = session.get(
            BASE_URL,
            params=params,
            timeout=15
        )
        response.raise_for_status()

        results = response.json().get("results", [])

        poster = next(
            (item.get("poster_path") for item in results
             if item.get("poster_path")),
            None
        )

        if poster:
            df.at[index, "poster_path"] = IMAGE_URL + poster
            success += 1
            print("Poster found!")
        else:
            failed += 1
            print("Poster unavailable.")

    except requests.exceptions.RequestException as error:
        failed += 1
        print("Request error:", error)

    # Save progress every 10 movies
    if count % 10 == 0:
        df.to_csv(CSV_PATH, index=False)
        print("Progress saved.")

    time.sleep(0.5)

df.to_csv(CSV_PATH, index=False)

print("\nTEST COMPLETED!")
print("Posters found:", success)
print("Not found/errors:", failed)
