
# ==========================================
# NLP MOVIE RECOMMENDATION SYSTEM
# STEP 3: FEATURE EXTRACTION
# ==========================================

import pandas as pd
import ast

# Step 1: Load cleaned dataset

movies = pd.read_csv("dataset/cleaned_movies.csv")


# Step 2: Convert JSON-like text into Python objects

def convert(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError):
        return []


# Step 3: Extract genres and keywords

def get_names(obj):
    data = convert(obj)

    return [
        item['name'].replace(" ", "")
        for item in data
        if isinstance(item, dict) and 'name' in item
    ]


movies['genres'] = movies['genres'].apply(get_names)
movies['keywords'] = movies['keywords'].apply(get_names)


# Step 4: Extract top 3 actors

def get_cast(obj):
    data = convert(obj)

    return [
        item['name'].replace(" ", "")
        for item in data[:3]
        if isinstance(item, dict) and 'name' in item
    ]


movies['cast'] = movies['cast'].apply(get_cast)


# Step 5: Extract director name

def get_director(obj):
    data = convert(obj)

    for item in data:
        if (
            isinstance(item, dict)
            and item.get('job') == 'Director'
        ):
            return [item['name'].replace(" ", "")]

    return []


movies['crew'] = movies['crew'].apply(get_director)


# Step 6: Process movie overview

movies['overview'] = (
    movies['overview']
    .fillna('')
    .apply(lambda x: x.split())
)


# Step 7: Combine all features into tags

movies['tags'] = (
    movies['overview']
    + movies['genres']
    + movies['keywords']
    + movies['cast']
    + movies['crew']
)

# Convert list into string

movies['tags'] = movies['tags'].apply(
    lambda x: " ".join(x).lower()
)


# Step 8: Create final dataset

new_movies = movies[['id', 'title', 'tags']].copy()


# Step 9: Display results

print("\n========== FEATURE EXTRACTION ==========")

print(new_movies.head())

print("\nTotal Movies:", len(new_movies))

print("\nExample Movie Tags:")
print(new_movies.iloc[0]['tags'])


# Step 10: Save the dataset

new_movies.to_csv(
    "dataset/movies_with_tags.csv",
    index=False
)

print("\n========================================")
print("FEATURE EXTRACTION COMPLETED!")
print("========================================")
