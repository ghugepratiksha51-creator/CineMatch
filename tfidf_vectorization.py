
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

# Load preprocessed movie dataset
movies = pd.read_csv("dataset/processed_movies.csv")

# Handle missing tags
movies['tags'] = movies['tags'].fillna('')

# Create TF-IDF object
tfidf = TfidfVectorizer(max_features=5000)

# Convert text into numerical vectors
tfidf_matrix = tfidf.fit_transform(movies['tags'])

# Display results
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
print("Total Movies:", len(movies))
print("Total Features:", len(tfidf.get_feature_names_out()))

# Save TF-IDF matrix and vectorizer
save_npz("dataset/tfidf_matrix.npz", tfidf_matrix)
joblib.dump(tfidf, "dataset/tfidf_vectorizer.pkl")

print("TF-IDF VECTORIZATION COMPLETED!")
