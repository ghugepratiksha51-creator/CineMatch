
import pandas as pd
from scipy.sparse import load_npz

# Load movie dataset
movies = pd.read_csv("dataset/processed_movies.csv")

# Load similarity matrix
similarity = load_npz("dataset/similarity_matrix.npz")

# Recommendation function
def recommend(movie_name):

    # Search for the movie
    movie_name = movie_name.strip().lower()

    matches = movies[
        movies['title'].str.lower() == movie_name
    ]

    if matches.empty:
        print("Movie not found! Please check the movie name.")
        return

    movie_index = matches.index[0]

    # Get similarity scores
    scores = similarity.getrow(movie_index).toarray().flatten()

    # Sort movies by similarity
    movie_indices = scores.argsort()[::-1]

    print("\nRecommended Movies for:", movies.iloc[movie_index]['title'])
    print("-----------------------------------")

    count = 0

    for index in movie_indices:
        if index == movie_index:
            continue

        print(movies.iloc[index]['title'])
        count += 1

        if count == 5:
            break

# Test the recommendation system
recommend("Avatar")
