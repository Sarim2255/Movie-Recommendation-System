import ast
import io
import pickle
import sys
from pathlib import Path

# Force UTF-8 output on Windows terminals that default to cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer   # ← TF-IDF (not CountVectorizer)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
BASE_DIR        = Path(__file__).resolve().parent
MOVIES_CSV      = BASE_DIR / "tmdb_5000_movies.csv"
CREDITS_CSV     = BASE_DIR / "tmdb_5000_credits.csv"
MOVIES_PKL      = BASE_DIR / "movies.pkl"
SIMILARITY_PKL  = BASE_DIR / "similarity.pkl"
CF_FACTORS_PKL  = BASE_DIR / "cf_factors.pkl"


# ---------------------------------------------------------------------------
# Utility parsers
# ---------------------------------------------------------------------------

def parse_names(value, limit=None, key="name"):
    """Parse a TMDB JSON-string column into a list of names (spaces removed).

    Spaces are removed so that multi-word names (e.g. 'Science Fiction') become
    single tokens ('ScienceFiction') in the TF-IDF vocabulary.
    """
    try:
        items = ast.literal_eval(value) if isinstance(value, str) else []
        names = [
            item[key].replace(" ", "")
            for item in items
            if isinstance(item, dict) and item.get(key)
        ]
        return names[:limit] if limit else names
    except (ValueError, SyntaxError, TypeError):
        return []


def parse_names_raw(value, limit=None, key="name"):
    """Parse a TMDB JSON-string column into a list of names (spaces preserved).

    Used for the collaborative filtering feature matrix where the original
    genre / keyword strings serve as categorical feature labels.
    """
    try:
        items = ast.literal_eval(value) if isinstance(value, str) else []
        names = [
            item[key]
            for item in items
            if isinstance(item, dict) and item.get(key)
        ]
        return names[:limit] if limit else names
    except (ValueError, SyntaxError, TypeError):
        return []


def get_director(value):
    """Extract the director's name from the crew JSON-string column."""
    try:
        crew = ast.literal_eval(value) if isinstance(value, str) else []
        for person in crew:
            if isinstance(person, dict) and person.get("job") == "Director":
                return [person.get("name", "").replace(" ", "")]
    except (ValueError, SyntaxError, TypeError):
        pass
    return []


# ---------------------------------------------------------------------------
# Model 1 — Content-Based Filtering (TF-IDF + Cosine Similarity)
# ---------------------------------------------------------------------------

def build_content_based_model(tags_series):
    """
    Fit a TF-IDF vectorizer on the combined movie tags and compute a full
    cosine similarity matrix.

    Parameters
    ----------
    tags_series : pd.Series
        One pre-processed tag string per movie.

    Returns
    -------
    similarity : np.ndarray, shape (n_movies, n_movies)
        Pairwise cosine similarity scores.
    """
    print("  Fitting TF-IDF vectorizer (max_features=5000, stop_words='english') ...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(tags_series)
    print(f"  TF-IDF matrix shape : {tfidf_matrix.shape}")

    print("  Computing cosine similarity matrix ...")
    similarity = cosine_similarity(tfidf_matrix)
    return similarity


# ---------------------------------------------------------------------------
# Model 2 — Collaborative Filtering (NMF Matrix Factorization)
# ---------------------------------------------------------------------------

def build_collaborative_filtering_model(genres_list, keywords_list):
    """
    Build item-based collaborative filtering via NMF matrix factorization.

    An implicit movie × feature matrix is constructed from genre and keyword
    labels (MultiLabelBinarizer).  NMF decomposes this into latent factors,
    analogous to discovering hidden user-preference dimensions.  The resulting
    per-movie latent vectors can be used to:

      * Find similar movies  : cosine_similarity(vec_A, all_vecs)
      * Build user profiles  : mean of watched-movies' latent vectors

    Parameters
    ----------
    genres_list   : list of list[str]  — raw genre names per movie
    keywords_list : list of list[str]  — raw keyword names per movie

    Returns
    -------
    latent_factors : np.ndarray, shape (n_movies, n_components)
    feature_names  : list[str]   — ordered feature labels used by the MLB
    """
    print("  Building implicit movie × feature matrix (genres + keywords) ...")
    # Merge and deduplicate features per movie
    combined = [sorted(set(g + k)) for g, k in zip(genres_list, keywords_list)]

    mlb = MultiLabelBinarizer()
    feature_matrix = mlb.fit_transform(combined).astype(np.float32)
    print(f"  Feature matrix : {feature_matrix.shape[0]} movies x {feature_matrix.shape[1]} features")

    n_components = min(50, feature_matrix.shape[1] - 1)
    print(f"  Fitting NMF (n_components={n_components}, init='nndsvda', max_iter=500) ...")
    nmf = NMF(
        n_components=n_components,
        init="nndsvda",
        random_state=42,
        max_iter=500,
        tol=1e-4,
    )
    latent_factors = nmf.fit_transform(feature_matrix)
    print(f"  Reconstruction error : {nmf.reconstruction_err_:.4f}")
    return latent_factors, mlb.classes_.tolist()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_model():
    """End-to-end pipeline: load → preprocess → build CB + CF → save."""

    # ------------------------------------------------------------------
    # 1. Validate inputs
    # ------------------------------------------------------------------
    for path in (MOVIES_CSV, CREDITS_CSV):
        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} not found.\n"
                "Place tmdb_5000_movies.csv and tmdb_5000_credits.csv "
                "in the same folder as main.py."
            )

    # ------------------------------------------------------------------
    # 2. Load & merge
    # ------------------------------------------------------------------
    print("Loading datasets ...")
    movies  = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)
    movies  = movies.merge(credits, on="title")

    id_col = "id" if "id" in movies.columns else "movie_id"
    required = [id_col, "title", "overview", "genres", "cast", "keywords", "crew"]
    missing  = [c for c in required if c not in movies.columns]
    if missing:
        raise ValueError(f"Dataset columns missing: {missing}")

    movies = movies[required].copy()
    movies.rename(columns={id_col: "movie_id"}, inplace=True)
    movies.dropna(subset=["overview"], inplace=True)
    movies = movies.reset_index(drop=True)
    print(f"Loaded {len(movies):,} movies after cleaning.")

    # ------------------------------------------------------------------
    # 3. Feature extraction
    # ------------------------------------------------------------------
    print("\nExtracting features ...")

    # Raw names (spaces preserved) — for CF features & UI display
    movies["genres_raw"]   = movies["genres"].apply(parse_names_raw)
    movies["keywords_raw"] = movies["keywords"].apply(parse_names_raw)

    # Processed names (spaces removed) — for TF-IDF tags
    movies["genres_proc"]   = movies["genres"].apply(parse_names)
    movies["keywords_proc"] = movies["keywords"].apply(parse_names)
    movies["cast_proc"]     = movies["cast"].apply(lambda x: parse_names(x, limit=3))
    movies["crew_proc"]     = movies["crew"].apply(get_director)
    movies["overview_proc"] = movies["overview"].astype(str).apply(str.split)

    # Combine all features into a single tag string
    movies["tags"] = (
        movies["overview_proc"]
        + movies["genres_proc"]
        + movies["keywords_proc"]
        + movies["cast_proc"]
        + movies["crew_proc"]
    )
    movies["tags"] = movies["tags"].apply(lambda ws: " ".join(ws).lower())

    # Porter Stemming — reduces vocabulary size, improves matching
    stemmer = PorterStemmer()
    movies["tags"] = movies["tags"].apply(
        lambda text: " ".join(stemmer.stem(w) for w in text.split())
    )

    # ------------------------------------------------------------------
    # 4. Build models
    # ------------------------------------------------------------------
    print("\n[1/2] Content-Based Filtering  --  TF-IDF -> Cosine Similarity")
    similarity = build_content_based_model(movies["tags"])

    print("\n[2/2] Collaborative Filtering  --  NMF Matrix Factorization")
    cf_latent_factors, cf_feature_names = build_collaborative_filtering_model(
        movies["genres_raw"].tolist(),
        movies["keywords_raw"].tolist(),
    )

    # ------------------------------------------------------------------
    # 5. Save artifacts
    # ------------------------------------------------------------------
    print("\nSaving model artifacts ...")

    save_movies = movies[["movie_id", "title", "genres_raw"]].copy()

    with MOVIES_PKL.open("wb") as f:
        pickle.dump(save_movies, f)

    with SIMILARITY_PKL.open("wb") as f:
        pickle.dump(similarity, f)

    with CF_FACTORS_PKL.open("wb") as f:
        pickle.dump({
            "factors":       cf_latent_factors,   # shape: (n_movies, n_components)
            "feature_names": cf_feature_names,     # ordered list of genre/keyword labels
            "n_movies":      len(save_movies),
        }, f)

    print(f"\nDone! Model built successfully.")
    print(f"   {MOVIES_PKL.name:<20}  {len(save_movies):,} movies")
    print(f"   {SIMILARITY_PKL.name:<20}  TF-IDF similarity  "
          f"({similarity.shape[0]}x{similarity.shape[1]})")
    print(f"   {CF_FACTORS_PKL.name:<20}  NMF latent factors "
          f"({cf_latent_factors.shape[0]}x{cf_latent_factors.shape[1]})")



if __name__ == "__main__":
    build_model()
