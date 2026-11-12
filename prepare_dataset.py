
# ==========================================
# NLP MOVIE RECOMMENDATION SYSTEM
# STEP 2: DATA CLEANING AND DATA MERGING
# ==========================================

import pandas as pd

# Step 1: Load both datasets

movies = pd.read_csv("dataset/tmdb_5000_movies.csv")
credits = pd.read_csv("dataset/tmdb_5000_credits.csv")

print("Original Movies Shape:", movies.shape)
print("Original Credits Shape:", credits.shape)


# Step 2: Select required columns from credits

credits = credits[['movie_id', 'cast', 'crew']]


# Step 3: Merge both datasets

credits = credits.rename(columns={'movie_id': 'id'})

movies = movies.merge(credits, on='id', how='inner')

print("\nAfter Merging:")
print("Merged Dataset Shape:", movies.shape)


# Step 4: Select important columns

movies = movies[
    [
        'id',
        'title',
        'overview',
        'genres',
        'keywords',
        'cast',
        'crew'
    ]
]


# Step 5: Handle missing values

movies['overview'] = movies['overview'].fillna('')
movies['genres'] = movies['genres'].fillna('[]')
movies['keywords'] = movies['keywords'].fillna('[]')
movies['cast'] = movies['cast'].fillna('[]')
movies['crew'] = movies['crew'].fillna('[]')

movies = movies.dropna(subset=['title'])


# Step 6: Remove duplicate movies

movies = movies.drop_duplicates(subset=['id'])


# Step 7: Reset dataset index

movies = movies.reset_index(drop=True)


# Step 8: Display cleaned dataset

print("\nCleaned Dataset:")
print(movies.head())


# Step 9: Check missing values

print("\nMissing Values:")
print(movies.isnull().sum())


# Step 10: Check final dataset shape

print("\nFinal Dataset Shape:")
print(movies.shape)


# Step 11: Save cleaned dataset

movies.to_csv(
    "dataset/cleaned_movies.csv",
    index=False
)

print("\n================================")
print("DATA CLEANING COMPLETED!")
print("Cleaned dataset saved successfully.")
print("================================")
