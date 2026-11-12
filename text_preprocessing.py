
# ==========================================
# NLP MOVIE RECOMMENDATION SYSTEM
# STEP 4: TEXT PREPROCESSING
# ==========================================

import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords
nltk.download('stopwords')

# Step 1: Load movie tags dataset

movies = pd.read_csv("dataset/movies_with_tags.csv")

# Step 2: Initialize stopwords and stemmer

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()


# Step 3: Create preprocessing function

def preprocess_text(text):

    # Convert to lowercase
    text = str(text).lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Tokenization
    words = text.split()

    # Remove stopwords
    words = [
        word for word in words
        if word not in stop_words
    ]

    # Apply stemming
    words = [
        ps.stem(word) for word in words
    ]

    # Join words
    return " ".join(words)


# Step 4: Apply preprocessing

movies['tags'] = movies['tags'].fillna('').apply(
    preprocess_text
)


# Step 5: Display results

print("\nOriginal Movie Tags:")
print(pd.read_csv(
    "dataset/movies_with_tags.csv"
)['tags'].head(3))

print("\nProcessed Movie Tags:")
print(movies['tags'].head(3))


# Step 6: Save processed dataset

movies.to_csv(
    "dataset/processed_movies.csv",
    index=False
)

print("\n================================")
print("TEXT PREPROCESSING COMPLETED!")
print("================================")
