import streamlit as st
import pandas as pd
import numpy as np
import requests
import re

from PIL import Image, ImageDraw
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CineMatch | Movie Recommendation",
    page_icon="🎬",
    layout="wide"
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset" / "combined_movies.csv"
RECOMMENDATIONS_PATH = BASE_DIR / "dataset" / "top_50_recommendations.npz"
BACKGROUND_PATH = BASE_DIR / "assets" / "cinematch_background.png"


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

def load_background():

    if BACKGROUND_PATH.exists():

        import base64

        with open(BACKGROUND_PATH, "rb") as file:
            encoded = base64.b64encode(file.read()).decode()

        return f"""
        background-image:
            linear-gradient(rgba(8, 10, 18, 0.88),
                            rgba(8, 10, 18, 0.94)),
            url("data:image/png;base64,{encoded}");
        """

    return "background-color: #080a12;"


background_css = load_background()

st.markdown(
    f"""
    <style>

    .stApp {{
        {background_css}
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }}

    .main-title {{
        color: #d8ad55;
        font-size: 36px;
        font-weight: 800;
        text-align: left;
        margin-bottom: 5px;
    }}

    .subtitle {{
        color: #d0d0d0;
        font-size: 15px;
        margin-bottom: 25px;
    }}

    .section-title {{
        color: #d8ad55;
        font-size: 23px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }}

    .movie-title {{
        color: #ffffff;
        font-size: 17px;
        font-weight: 700;
        margin-top: 10px;
        min-height: 45px;
    }}

    .movie-info {{
        color: #cccccc;
        font-size: 13px;
    }}

    div.stButton > button {{
        background-color: #d8ad55;
        color: #111111;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        padding: 10px 20px;
    }}

    div.stButton > button:hover {{
        background-color: #f0c96e;
        color: #111111;
    }}

    div[data-testid="stSelectbox"] label,
    div[data-testid="stRadio"] label {{
        color: white;
    }}

    div[data-testid="stMarkdownContainer"] p {{
        color: #eeeeee;
    }}

    .footer {{
        text-align: center;
        color: #aaaaaa;
        font-size: 12px;
        margin-top: 45px;
        padding: 15px;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# LOAD DATASET AND COMPRESSED RECOMMENDATIONS
# --------------------------------------------------

@st.cache_data
def load_dataset():

    df = pd.read_csv(DATASET_PATH)
    df = df.reset_index(drop=True)

    return df


@st.cache_resource
def load_recommendations():

    with np.load(RECOMMENDATIONS_PATH) as data:

        return {
            key: data[key]
            for key in data.files
        }


try:

    movies = load_dataset()
    recommendations_data = load_recommendations()

except Exception as error:

    st.error(f"Error loading project files: {error}")
    st.stop()


# Check dataset and recommendation alignment
if len(movies) != recommendations_data["all_indices"].shape[0]:

    st.error(
        "Dataset and recommendation file do not match. "
        "Please regenerate the compressed recommendations."
    )

    st.stop()


# --------------------------------------------------
# POSTER URL HANDLING
# --------------------------------------------------

def get_poster(row):

    poster_url = row.get("poster_path")

    if pd.isna(poster_url) or not str(poster_url).strip():
        return None

    poster_url = str(poster_url).strip()

    # Existing complete URLs
    if poster_url.startswith("http://") or poster_url.startswith("https://"):
        return poster_url

    # TMDB image paths
    if poster_url.startswith("/"):
        return "https://image.tmdb.org/t/p/w500" + poster_url

    return None


# --------------------------------------------------
# CONVERT WIKIMEDIA THUMBNAIL TO ORIGINAL URL
# --------------------------------------------------

def convert_wikimedia_url(poster_url):

    if "upload.wikimedia.org" not in poster_url:
        return poster_url

    parts = urlsplit(poster_url)
    path_parts = parts.path.split("/")

    if "thumb" in path_parts:

        thumb_index = path_parts.index("thumb")

        original_path = (
            path_parts[:thumb_index]
            + path_parts[thumb_index + 1:-1]
        )

        original_url = urlunsplit((
            parts.scheme,
            parts.netloc,
            "/".join(original_path),
            parts.query,
            parts.fragment
        ))

        return original_url

    return poster_url


# --------------------------------------------------
# CREATE PLACEHOLDER POSTER
# --------------------------------------------------

def create_placeholder():

    image = Image.new(
        "RGB",
        (300, 450),
        color=(23, 27, 38)
    )

    draw = ImageDraw.Draw(image)

    text = "POSTER\nUNAVAILABLE"

    draw.multiline_text(
        (150, 215),
        text,
        fill=(216, 173, 85),
        anchor="mm",
        align="center",
        spacing=12
    )

    return image


# --------------------------------------------------
# DOWNLOAD AND LOAD POSTER
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def load_poster_image(poster_url):

    try:

        if not poster_url:
            return None

        # Convert Wikimedia thumbnail URLs
        poster_url = convert_wikimedia_url(poster_url)

        headers = {
            "User-Agent": "CineMatch/1.0 (educational movie recommendation project)"
        }

        if "wikimedia.org" in poster_url:
            headers["Referer"] = "https://en.wikipedia.org/"

        response = requests.get(
            poster_url,
            headers=headers,
            timeout=20
        )

        response.raise_for_status()

        image = Image.open(
            BytesIO(response.content)
        ).convert("RGB")

        return image

    except Exception:
        return None


# --------------------------------------------------
# DISPLAY POSTER
# --------------------------------------------------

def display_poster(poster_url):

    image = load_poster_image(poster_url)

    if image is not None:
        st.image(image, width="stretch")

    else:
        st.image(create_placeholder(), width="stretch")


# --------------------------------------------------
# GENRE HELPERS
# --------------------------------------------------

def parse_genres(value):

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value or value == r"\N":
        return []

    # Keep Science Fiction as one genre if it appears in the dataset.
    value = re.sub(
        r"\bScience Fiction\b",
        "Science_Fiction",
        value,
        flags=re.IGNORECASE
    )

    # Dataset values use both spaces and pipe separators.
    genre_values = re.split(r"[|\s]+", value)

    parsed = []

    for genre_name in genre_values:

        if not genre_name or genre_name == r"\N":
            continue

        # Movie and TV are content types, not genres.
        if genre_name.lower() in ["movie", "tv"]:
            continue

        if genre_name == "Science_Fiction":
            genre_name = "Science Fiction"

        parsed.append(genre_name)

    return parsed


# --------------------------------------------------
# RECOMMENDATION FUNCTION
# --------------------------------------------------

def recommend_movies(
    selected_index,
    category="All Movies",
    selected_genre="All Genres",
    top_n=5
):

    if category == "Bollywood":

        key = "bollywood_indices"

    elif category == "Hollywood":

        key = "hollywood_indices"

    else:

        key = "all_indices"

    # Get precomputed top 50 movie indexes
    recommended_indexes = recommendations_data[key][selected_index]

    # Exclude selected movie and invalid indexes
    recommended_indexes = [
        int(index)
        for index in recommended_indexes
        if 0 <= int(index) < len(movies)
        and int(index) != selected_index
    ]

    # Keep recommendations in their original similarity-ranked order.
    recommended_movies = movies.iloc[recommended_indexes]

    # Apply the selected genre to the ranked recommendations.
    if selected_genre != "All Genres":
        recommended_movies = recommended_movies[
            recommended_movies["genres"].apply(
                lambda value: selected_genre in parse_genres(value)
            )
        ]

    return recommended_movies.head(top_n)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🎬 CineMatch</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Discover movies you will love with intelligent movie recommendations.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# CATEGORY FILTER
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Explore Movies</div>',
    unsafe_allow_html=True
)

category = st.radio(
    "Select Movie Category",
    ["All Movies", "Bollywood", "Hollywood"],
    horizontal=True
)

# Build clean genre choices from the dataset.
available_genres = sorted({
    genre_name
    for value in movies["genres"].dropna()
    for genre_name in parse_genres(value)
})

selected_genre = st.selectbox(
    "Select Movie Genre",
    ["All Genres"] + available_genres
)


# --------------------------------------------------
# FILTER MOVIES
# --------------------------------------------------

if category == "All Movies":

    filtered_movies = movies.copy()

else:

    filtered_movies = movies[
        movies["category"].astype(str).str.lower()
        == category.lower()
    ].copy()

# Apply genre filter to the movie selection list.
if selected_genre != "All Genres":
    filtered_movies = filtered_movies[
        filtered_movies["genres"].apply(
            lambda value: selected_genre in parse_genres(value)
        )
    ]


if filtered_movies.empty:

    st.warning("No movies available for this category and genre.")
    st.stop()


# --------------------------------------------------
# MOVIE SELECTION
# --------------------------------------------------

movie_options = filtered_movies["title"].fillna(
    "Unknown Movie"
).tolist()

selected_movie = st.selectbox(
    "Choose a movie you like",
    movie_options
)


# Get original dataset index
selected_index = filtered_movies[
    filtered_movies["title"].fillna("Unknown Movie")
    == selected_movie
].index[0]


# --------------------------------------------------
# RECOMMENDATION BUTTON
# --------------------------------------------------

if st.button("✨ Recommend Movies", use_container_width=True):

    st.session_state["selected_movie"] = selected_movie
    st.session_state["selected_index"] = selected_index
    st.session_state["selected_category"] = category
    st.session_state["selected_genre"] = selected_genre


# --------------------------------------------------
# DISPLAY RECOMMENDATIONS
# --------------------------------------------------

if "selected_index" in st.session_state:

    selected_index = st.session_state["selected_index"]
    selected_movie = st.session_state["selected_movie"]
    selected_category = st.session_state["selected_category"]
    selected_genre = st.session_state.get("selected_genre", "All Genres")

    st.markdown(
        '<div class="section-title">🎥 Recommended Movies For You</div>',
        unsafe_allow_html=True
    )

    recommendations = recommend_movies(
        selected_index,
        selected_category,
        selected_genre=selected_genre,
        top_n=5
    )

    if recommendations.empty:

        st.info(
            "No recommendations found for this genre. "
            "Try selecting All Genres or another genre."
        )

    else:

        columns = st.columns(5)

        for position, (_, movie) in enumerate(
            recommendations.iterrows()
        ):

            with columns[position]:

                poster_url = get_poster(movie)

                display_poster(poster_url)

                st.markdown(
                    f'<div class="movie-title">{movie["title"]}</div>',
                    unsafe_allow_html=True
                )

                release_date = movie.get("release_date", "N/A")

                if pd.isna(release_date):
                    release_date = "N/A"

                st.markdown(
                    f'<div class="movie-info">'
                    f'📅 {release_date}</div>',
                    unsafe_allow_html=True
                )

                rating = movie.get("vote_average", "N/A")

                if pd.isna(rating):
                    rating = "N/A"

                st.markdown(
                    f'<div class="movie-info">'
                    f'⭐ {rating}</div>',
                    unsafe_allow_html=True
                )

                with st.expander("More Details"):

                    st.write(
                        "**Genre:**",
                        movie.get("genres", "Not available")
                    )

                    st.write(
                        "**Director:**",
                        movie.get("director", "Not available")
                    )

                    st.write(
                        "**Cast:**",
                        movie.get("cast", "Not available")
                    )

                    st.write(
                        "**Overview:**",
                        movie.get("overview", "No overview available")
                    )


# --------------------------------------------------
# SELECTED MOVIE DETAILS
# --------------------------------------------------

if "selected_index" in st.session_state:

    selected_row = movies.iloc[
        st.session_state["selected_index"]
    ]

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Selected Movie Details</div>',
        unsafe_allow_html=True
    )

    detail_col1, detail_col2 = st.columns([1, 3])

    with detail_col1:

        display_poster(get_poster(selected_row))

    with detail_col2:

        st.subheader(selected_row.get("title", "Unknown Movie"))

        st.write(
            "**Category:**",
            selected_row.get("category", "N/A")
        )

        st.write(
            "**Genres:**",
            selected_row.get("genres", "N/A")
        )

        st.write(
            "**Director:**",
            selected_row.get("director", "N/A")
        )

        st.write(
            "**Cast:**",
            selected_row.get("cast", "N/A")
        )

        st.write(
            "**Release Date:**",
            selected_row.get("release_date", "N/A")
        )

        st.write(
            "**Rating:**",
            selected_row.get("vote_average", "N/A")
        )

        st.write(
            "**Overview:**",
            selected_row.get("overview", "No overview available")
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        🎬 CineMatch | Movie Recommendation System
        <br>
        Built using Python, Streamlit, TF-IDF and Cosine Similarity
    </div>
    """,
    unsafe_allow_html=True
)
