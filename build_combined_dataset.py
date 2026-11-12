
import pandas as pd
import json
import re
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz


# ==========================================
# 1. PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

HOLLYWOOD_MOVIES = DATASET_DIR / "tmdb_5000_movies.csv"
HOLLYWOOD_CREDITS = DATASET_DIR / "tmdb_5000_credits.csv"
BOLLYWOOD_MOVIES = DATASET_DIR / "bollywood_full.csv"


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def parse_json(value):

    if not isinstance(value, str):
        return []

    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


def clean_text(value):

    if pd.isna(value):
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()


def get_names(value, limit=None):

    items = parse_json(value)

    names = [
        item.get("name", "")
        for item in items
        if isinstance(item, dict) and item.get("name")
    ]

    if limit:
        names = names[:limit]

    return " ".join(names)


def get_director(value):

    crew = parse_json(value)

    for person in crew:
        if (
            isinstance(person, dict)
            and person.get("job") == "Director"
        ):
            return person.get("name", "")

    return ""


# ==========================================
# 3. LOAD HOLLYWOOD DATA
# ==========================================

print("Loading Hollywood dataset...")

hollywood = pd.read_csv(HOLLYWOOD_MOVIES)
credits = pd.read_csv(HOLLYWOOD_CREDITS)

if "id" not in hollywood.columns:
    raise ValueError("TMDB movies CSV must contain an id column.")

if "movie_id" in credits.columns:
    credits = credits.rename(columns={"movie_id": "id"})

if "id" not in credits.columns:
    raise ValueError("TMDB credits CSV must contain movie_id or id.")

# Keep only the columns needed from credits
credit_columns = ["id", "cast", "crew"]

credits = credits[
    [col for col in credit_columns if col in credits.columns]
].copy()

# Merge without duplicating the title column
hollywood = hollywood.merge(
    credits,
    on="id",
    how="left"
)

hollywood["title"] = hollywood["title"].fillna("")

hollywood["overview"] = (
    hollywood["overview"]
    if "overview" in hollywood.columns
    else ""
)

hollywood["genres_text"] = hollywood["genres"].apply(
    get_names
) if "genres" in hollywood.columns else ""

hollywood["cast_text"] = hollywood["cast"].apply(
    lambda x: get_names(x, limit=5)
) if "cast" in hollywood.columns else ""

hollywood["director_text"] = hollywood["crew"].apply(
    get_director
) if "crew" in hollywood.columns else ""

hollywood["poster_path"] = (
    hollywood["poster_path"]
    if "poster_path" in hollywood.columns
    else ""
)

hollywood["release_date"] = (
    hollywood["release_date"]
    if "release_date" in hollywood.columns
    else ""
)

hollywood["vote_average"] = (
    hollywood["vote_average"]
    if "vote_average" in hollywood.columns
    else 0
)

hollywood_final = pd.DataFrame({
    "title": hollywood["title"],
    "overview": hollywood["overview"],
    "genres": hollywood["genres_text"],
    "cast": hollywood["cast_text"],
    "director": hollywood["director_text"],
    "poster_path": hollywood["poster_path"],
    "release_date": hollywood["release_date"],
    "vote_average": hollywood["vote_average"],
    "category": "Hollywood"
})


# ==========================================
# 4. LOAD BOLLYWOOD DATA
# ==========================================

print("Loading Bollywood dataset...")

bollywood = pd.read_csv(BOLLYWOOD_MOVIES)

title_column = (
    "title_x" if "title_x" in bollywood.columns
    else "title_y" if "title_y" in bollywood.columns
    else "original_title"
)

bollywood_title = bollywood[title_column]

if "title_y" in bollywood.columns:
    bollywood_title = bollywood_title.fillna(
        bollywood["title_y"]
    )

if "original_title" in bollywood.columns:
    bollywood_title = bollywood_title.fillna(
        bollywood["original_title"]
    )

bollywood_final = pd.DataFrame({
    "title": bollywood_title,
    "overview": (
        bollywood["summary"].fillna("")
        if "summary" in bollywood.columns
        else ""
    ),
    "genres": (
        bollywood["genres"].fillna("").astype(str)
        if "genres" in bollywood.columns
        else ""
    ),
    "cast": (
        bollywood["actors"].fillna("").astype(str)
        if "actors" in bollywood.columns
        else ""
    ),
    "director": "",
    "poster_path": (
        bollywood["poster_path"].fillna("")
        if "poster_path" in bollywood.columns
        else ""
    ),
    "release_date": (
        bollywood["year_of_release"].fillna("")
        if "year_of_release" in bollywood.columns
        else ""
    ),
    "vote_average": (
        bollywood["imdb_rating"].fillna(0)
        if "imdb_rating" in bollywood.columns
        else 0
    ),
    "category": "Bollywood"
})

# Add story to the overview for better recommendations
if "story" in bollywood.columns:
    bollywood_final["overview"] = (
        bollywood_final["overview"].astype(str)
        + " "
        + bollywood["story"].fillna("").astype(str)
    )


# ==========================================
# 5. COMBINE BOTH DATASETS
# ==========================================

print("Combining movies...")

movies = pd.concat(
    [hollywood_final, bollywood_final],
    ignore_index=True
)

movies["title"] = movies["title"].apply(clean_text)

movies = movies[
    movies["title"].ne("")
].copy()

# Remove duplicate movies within each category
movies = movies.drop_duplicates(
    subset=["title", "category"],
    keep="first"
).reset_index(drop=True)


# ==========================================
# 6. CREATE NLP TAGS
# ==========================================

print("Preparing movie text...")

for column in [
    "overview",
    "genres",
    "cast",
    "director"
]:
    movies[column] = movies[column].apply(clean_text)

movies["tags"] = (
    movies["overview"] + " "
    + movies["genres"] + " "
    + movies["cast"] + " "
    + movies["director"]
).str.lower()

movies["tags"] = movies["tags"].str.replace(
    r"[^a-zA-Z0-9\s]",
    " ",
    regex=True
)

movies["tags"] = movies["tags"].str.replace(
    r"\s+",
    " ",
    regex=True
).str.strip()

movies = movies.reset_index(drop=True)


# ==========================================
# 7. TF-IDF VECTORIZATION
# ==========================================

print("Creating TF-IDF vectors...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(
    movies["tags"]
)


# ==========================================
# 8. COSINE SIMILARITY
# ==========================================

print("Calculating cosine similarity...")

similarity_matrix = cosine_similarity(
    tfidf_matrix,
    dense_output=False
)


# ==========================================
# 9. SAVE OUTPUT FILES
# ==========================================

print("Saving combined dataset and model files...")

movies.to_csv(
    DATASET_DIR / "combined_movies.csv",
    index=False
)

# Save under the filenames used by the existing project
movies.to_csv(
    DATASET_DIR / "processed_movies.csv",
    index=False
)

save_npz(
    DATASET_DIR / "tfidf_matrix.npz",
    tfidf_matrix
)

save_npz(
    DATASET_DIR / "similarity_matrix.npz",
    similarity_matrix
)

joblib.dump(
    vectorizer,
    DATASET_DIR / "tfidf_vectorizer.pkl"
)


# ==========================================
# 10. FINAL REPORT
# ==========================================

print("\n================================")
print("CINEMATCH DATASET CREATED!")
print("================================")

print("Total movies:", len(movies))

print(
    "Bollywood movies:",
    (movies["category"] == "Bollywood").sum()
)

print(
    "Hollywood movies:",
    (movies["category"] == "Hollywood").sum()
)

print("\nSaved files:")
print("- combined_movies.csv")
print("- processed_movies.csv")
print("- tfidf_matrix.npz")
print("- similarity_matrix.npz")
print("- tfidf_vectorizer.pkl")

print("\nDataset integration completed successfully!")
