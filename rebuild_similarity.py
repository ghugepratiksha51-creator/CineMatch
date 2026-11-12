
import pandas as pd
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz

DATASET_DIR = Path("dataset")
CSV_PATH = DATASET_DIR / "combined_movies.csv"

print("Loading combined movie dataset...")

movies = pd.read_csv(CSV_PATH)

if "tags" not in movies.columns:
    raise ValueError("The dataset does not contain a tags column.")

movies["tags"] = movies["tags"].fillna("").astype(str)

print("Total movies:", len(movies))

print("Creating TF-IDF vectors...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(movies["tags"])

print("TF-IDF shape:", tfidf_matrix.shape)

print("Calculating cosine similarity...")

similarity_matrix = cosine_similarity(
    tfidf_matrix,
    dense_output=False
)

print("Similarity shape:", similarity_matrix.shape)

print("Saving updated model files...")

save_npz(DATASET_DIR / "tfidf_matrix.npz", tfidf_matrix)
save_npz(DATASET_DIR / "similarity_matrix.npz", similarity_matrix)

joblib.dump(vectorizer, DATASET_DIR / "tfidf_vectorizer.pkl")

movies.to_csv(DATASET_DIR / "processed_movies.csv", index=False)

print("\nMATRIX REBUILD COMPLETED!")
print("Dataset rows:", len(movies))
print("TF-IDF rows:", tfidf_matrix.shape[0])
print("Similarity matrix:", similarity_matrix.shape)
print("Original combined_movies.csv and its poster URLs are preserved.")
