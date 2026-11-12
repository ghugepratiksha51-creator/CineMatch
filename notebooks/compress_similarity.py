
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os

print("Loading movie dataset...")
movies = pd.read_csv("dataset/combined_movies.csv")

print("Loading similarity matrix...")
similarity = sp.load_npz("dataset/similarity_matrix.npz").tocsr()

TOP_K = 50
TOTAL_MOVIES = len(movies)

categories = {
    "all": np.arange(TOTAL_MOVIES),
    "bollywood": np.where(
        movies["category"].str.lower().values == "bollywood"
    )[0],
    "hollywood": np.where(
        movies["category"].str.lower().values == "hollywood"
    )[0]
}

results = {}

for category_name, candidate_indices in categories.items():

    print(f"\nProcessing {category_name}...")

    top_indices = np.zeros((TOTAL_MOVIES, TOP_K), dtype=np.int32)
    top_scores = np.zeros((TOTAL_MOVIES, TOP_K), dtype=np.float32)

    for movie_index in range(TOTAL_MOVIES):

        if movie_index % 1000 == 0:
            print(f"Processed {movie_index}/{TOTAL_MOVIES}")

        row = similarity.getrow(movie_index)

        # Start with zero similarity for every movie.
        scores = np.zeros(TOTAL_MOVIES, dtype=np.float32)

        scores[row.indices] = row.data.astype(np.float32)

        # Exclude the selected movie.
        scores[movie_index] = -1

        # Keep only the requested category.
        category_scores = scores[candidate_indices]

        count = min(TOP_K, len(candidate_indices))

        # Select highest-scoring movies.
        if count < len(category_scores):
            selected_positions = np.argpartition(
                category_scores, -count
            )[-count:]
        else:
            selected_positions = np.arange(count)

        selected_positions = selected_positions[
            np.argsort(category_scores[selected_positions])[::-1]
        ]

        selected_movies = candidate_indices[selected_positions]

        top_indices[movie_index, :count] = selected_movies
        top_scores[movie_index, :count] = scores[selected_movies]

    results[f"{category_name}_indices"] = top_indices
    results[f"{category_name}_scores"] = top_scores

print("\nSaving compressed recommendations...")

np.savez_compressed(
    "dataset/top_50_recommendations.npz",
    **results
)

file_size = os.path.getsize(
    "dataset/top_50_recommendations.npz"
) / (1024 * 1024)

print("\nSUCCESS!")
print("Total movies:", TOTAL_MOVIES)
print("Recommendations per category:", TOP_K)
print("New file size:", round(file_size, 2), "MB")
print("Saved: dataset/top_50_recommendations.npz")
