
import pandas as pd
import joblib
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

# Load movie dataset
movies = pd.read_csv("dataset/processed_movies.csv")

# Load TF-IDF matrix
tfidf_matrix = load_npz("dataset/tfidf_matrix.npz")

# Calculate similarity between movies
similarity_matrix = cosine_similarity(tfidf_matrix, dense_output=False)

# Display results
print("Similarity Matrix Shape:", similarity_matrix.shape)
print("Total Movies:", len(movies))

# Check similarity between first two movies
print("Similarity Score:", similarity_matrix[0, 1])

# Save similarity matrix
save_path = "dataset/similarity_matrix.npz"

from scipy.sparse import save_npz
save_npz(save_path, similarity_matrix)

print("COSINE SIMILARITY COMPLETED!")
