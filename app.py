"""
app.py — CineMatch | AI Movie Recommendation System
=====================================================
Streamlit frontend for the Movie Recommendation System.

Recommendation Modes
--------------------
  Content-Based   : TF-IDF cosine similarity on movie tags
  Collaborative   : NMF latent-factor cosine similarity (item-based CF)
  Hybrid          : Weighted blend — 60 % content + 40 % collaborative
  Personalized    : User watch-history drives a CF preference vector
                    (40 % content + 30 % CF + 30 % user preference)

User Watch History
------------------
  Users mark movies as watched via the "+ Watched" button.
  The watch history is stored in st.session_state and used to:
    • Build a personalized "For You" section
    • Visualise the user's genre taste profile in the sidebar
    • Enable the "Personalized" recommendation mode
"""

import os
import pickle
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Page config — MUST be the very first Streamlit call
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="CineMatch — AI Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Premium dark cinematic theme
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
/* ── Google Font ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*, *::before, *::after { font-family: 'Inter', sans-serif; box-sizing: border-box; }

/* ── App background ──────────────────────────────────────────────────── */
.stApp {
    background: radial-gradient(ellipse at 20% 0%, #1a0533 0%, #0a0a12 55%, #0a0a12 100%);
    min-height: 100vh;
}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #110d1f 0%, #0d0d14 100%) !important;
    border-right: 1px solid rgba(139, 92, 246, 0.15) !important;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* ── Main header ─────────────────────────────────────────────────────── */
.cinch-header {
    text-align: center;
    padding: 2rem 0 1rem;
}
.cinch-header h1 {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #c084fc 0%, #f59e0b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem;
}
.cinch-header p {
    color: #64748b;
    font-size: 0.95rem;
    margin: 0;
}

/* ── Stat cards ──────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(139, 92, 246, 0.35);
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: #64748b !important; font-size: 0.8rem !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e2e8f0 !important; font-size: 1.5rem !important; font-weight: 700 !important;
}

/* ── Section headers ─────────────────────────────────────────────────── */
.section-title {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 1.75rem 0 1rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.section-title h3 {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0;
}
.section-title .badge {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
}
.badge-cb   { background: rgba(139,92,246,0.18); color: #c084fc; }
.badge-cf   { background: rgba(245,158,11,0.18); color: #fbbf24; }
.badge-hyb  { background: rgba(16,185,129,0.18); color: #34d399; }
.badge-pers { background: rgba(236,72,153,0.18); color: #f472b6; }

/* ── Movie card wrapper ───────────────────────────────────────────────── */
.movie-wrap {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 0.75rem 0.6rem 0.9rem;
    transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
}
.movie-wrap:hover {
    transform: translateY(-5px);
    border-color: rgba(139, 92, 246, 0.45);
    box-shadow: 0 8px 24px rgba(139, 92, 246, 0.12);
}
.movie-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: #e2e8f0;
    line-height: 1.35;
    margin: 0.45rem 0 0.2rem;
    min-height: 2.2em;
}
.score-label {
    font-size: 0.68rem;
    color: #64748b;
    margin-bottom: 3px;
}
.score-track {
    height: 3px;
    background: rgba(255,255,255,0.08);
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 0.4rem;
}
.score-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #8b5cf6, #f59e0b);
    transition: width 0.6s ease;
}
.genre-tags {
    font-size: 0.65rem;
    color: #475569;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 0.35rem;
}
.in-history-label {
    font-size: 0.68rem;
    color: #34d399;
}

/* ── Buttons ─────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #9333ea 100%);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    transition: transform 0.15s, box-shadow 0.15s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(124, 58, 237, 0.45);
}
.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #94a3b8 !important;
}

/* ── Selectbox / inputs ──────────────────────────────────────────────── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ── Divider ─────────────────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* ── Info / warning boxes ────────────────────────────────────────────── */
.stAlert { border-radius: 10px !important; }

/* ── Spinner ─────────────────────────────────────────────────────────── */
.stSpinner { color: #8b5cf6 !important; }

/* ── History bar ─────────────────────────────────────────────────────── */
.hist-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.3rem 0.5rem;
    border-radius: 6px;
    margin-bottom: 0.2rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.05);
}
.hist-row span {
    font-size: 0.75rem;
    color: #94a3b8;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 155px;
}

/* ── Taste profile bar ───────────────────────────────────────────────── */
.taste-row { margin-bottom: 7px; }
.taste-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: #64748b;
    margin-bottom: 2px;
}
.taste-track {
    height: 4px;
    background: rgba(255,255,255,0.07);
    border-radius: 2px;
    overflow: hidden;
}
.taste-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #8b5cf6, #f59e0b);
}

/* ── Footer ──────────────────────────────────────────────────────────── */
.footer-text {
    text-align: center;
    color: #374151;
    font-size: 0.75rem;
    padding: 0.5rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "watch_history" not in st.session_state:
    st.session_state.watch_history = []  # list of dicts: {title, movie_id, genres, index}

# ---------------------------------------------------------------------------
# TMDB poster fetching
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_poster_api(movie_id: int, api_key: str):
    """Cached TMDB API call that returns the poster URL or None."""
    resp = requests.get(
        f"https://api.themoviedb.org/3/movie/{movie_id}",
        params={"api_key": api_key, "language": "en-US"},
        timeout=(5, 12),
    )
    if resp.status_code == 401:
        raise ValueError("Invalid TMDB API Key!")
    if resp.status_code == 404:
        raise ValueError("Movie not found on TMDB!")
    resp.raise_for_status()
    path = resp.json().get("poster_path")
    return f"https://image.tmdb.org/t/p/w500{path}" if path else None


def fetch_poster(movie_id: int):
    """Fetch a movie poster URL with retry logic.  Returns (url, error_msg)."""
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        return None, "TMDB API key missing — set TMDB_API_KEY in .env"
    for attempt in range(3):
        try:
            url = _fetch_poster_api(int(movie_id), api_key)
            return (url, None) if url else (None, "Poster not available on TMDB.")
        except ValueError as exc:
            return None, str(exc)
        except requests.RequestException:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
        except Exception:
            return None, "Unable to fetch poster."
    return None, "TMDB connection failed after 3 attempts."


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading recommendation models …")
def load_models():
    """Load movies metadata, TF-IDF similarity matrix, and CF latent factors."""
    movies_path    = BASE_DIR / "movies.pkl"
    sim_path       = BASE_DIR / "similarity.pkl"
    cf_path        = BASE_DIR / "cf_factors.pkl"

    for p in (movies_path, sim_path):
        if not p.exists():
            raise FileNotFoundError(f"{p.name} not found!  Run main.py first.")

    with open(movies_path, "rb") as f:
        movies = pickle.load(f)
    with open(sim_path, "rb") as f:
        similarity = pickle.load(f)

    movies = movies.reset_index(drop=True)
    if len(movies) != similarity.shape[0]:
        raise ValueError("movies.pkl and similarity.pkl are mismatched — re-run main.py.")

    cf_data = None
    if cf_path.exists():
        with open(cf_path, "rb") as f:
            cf_data = pickle.load(f)

    # Ensure genres_raw column exists (backwards-compat with old movies.pkl)
    if "genres_raw" not in movies.columns:
        movies["genres_raw"] = [[] for _ in range(len(movies))]

    return movies, similarity, cf_data


try:
    movies, similarity, cf_data = load_models()
except Exception as exc:
    st.error(f"⚠️  Model loading error: {exc}")
    st.stop()

has_cf = cf_data is not None

# Determine which column holds the TMDB movie ID
id_col = (
    "movie_id" if "movie_id" in movies.columns
    else "id" if "id" in movies.columns
    else None
)
if id_col is None:
    st.error("Movie ID column not found in movies.pkl!")
    st.stop()

# Collect all unique genres for the filter widget
all_genres: list[str] = sorted(
    {
        genre
        for genres in movies["genres_raw"]
        for genre in (genres if isinstance(genres, list) else [])
    }
)


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------

def _content_scores(movie_index: int) -> np.ndarray:
    """Return TF-IDF cosine similarity scores (content-based) for all movies."""
    return similarity[movie_index].copy()


def _cf_scores(movie_index: int):
    """Return NMF latent-factor cosine similarity scores (collaborative) for all movies."""
    if not has_cf:
        return None
    factors = cf_data["factors"]
    vec = factors[movie_index].reshape(1, -1)
    return cosine_similarity(vec, factors)[0]


def _user_preference_scores():
    """
    Build a user preference vector from the watch history (mean of watched
    movies' NMF latent factors) and return cosine similarity to all movies.
    """
    if not has_cf or not st.session_state.watch_history:
        return None
    factors = cf_data["factors"]
    indices = [
        h["index"] for h in st.session_state.watch_history
        if h["index"] < len(factors)
    ]
    if not indices:
        return None
    user_vec = np.mean(factors[indices], axis=0, keepdims=True)
    return cosine_similarity(user_vec, factors)[0]


def recommend(
    selected_movie: str,
    mode: str,
    top_n: int,
    genre_filter: list | None = None,
) -> list[dict]:
    """
    Return top-N movie recommendations.

    Parameters
    ----------
    selected_movie : movie title string (must exist in movies["title"])
    mode           : "Content-Based" | "Collaborative" | "Hybrid" | "Personalized"
    top_n          : number of results to return
    genre_filter   : optional list of genre strings to restrict results to

    Returns
    -------
    list of dicts with keys: title, movie_id, score, genres, index, in_history
    """
    matches = movies.index[movies["title"] == selected_movie].tolist()
    if not matches:
        return []
    movie_index = matches[0]

    cb  = _content_scores(movie_index)
    cf  = _cf_scores(movie_index)
    pref = _user_preference_scores()

    if mode == "Content-Based" or cf is None:
        scores = cb
    elif mode == "Collaborative":
        scores = cf
    elif mode == "Hybrid":
        scores = 0.60 * cb + 0.40 * cf
    elif mode == "Personalized":
        if pref is not None and cf is not None:
            scores = 0.40 * cb + 0.30 * cf + 0.30 * pref
        elif cf is not None:
            scores = 0.60 * cb + 0.40 * cf
        else:
            scores = cb
    else:
        scores = cb

    watched_titles = {h["title"] for h in st.session_state.watch_history}
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in ranked:
        if idx == movie_index:
            continue
        row = movies.iloc[idx]
        genres = row["genres_raw"] if isinstance(row["genres_raw"], list) else []

        if genre_filter and not any(g in genres for g in genre_filter):
            continue

        results.append({
            "title":      row["title"],
            "movie_id":   int(row[id_col]),
            "score":      float(score),
            "genres":     genres,
            "index":      int(idx),
            "in_history": row["title"] in watched_titles,
        })
        if len(results) == top_n:
            break
    return results


def personalized_for_you(top_n: int = 5) -> list[dict]:
    """Return personalized recommendations driven purely by watch history."""
    if not has_cf or not st.session_state.watch_history:
        return []
    pref = _user_preference_scores()
    if pref is None:
        return []

    watched_titles = {h["title"] for h in st.session_state.watch_history}
    watched_idx    = {h["index"] for h in st.session_state.watch_history}
    ranked = sorted(enumerate(pref), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in ranked:
        if idx in watched_idx:
            continue
        row = movies.iloc[idx]
        if row["title"] in watched_titles:
            continue
        genres = row["genres_raw"] if isinstance(row["genres_raw"], list) else []
        results.append({
            "title":      row["title"],
            "movie_id":   int(row[id_col]),
            "score":      float(score),
            "genres":     genres,
            "index":      int(idx),
            "in_history": False,
        })
        if len(results) == top_n:
            break
    return results


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _add_to_history(title: str, movie_id: int, genres: list, index: int):
    """Add a movie to the session watch history (no-op if already present)."""
    if not any(h["title"] == title for h in st.session_state.watch_history):
        st.session_state.watch_history.append(
            {"title": title, "movie_id": movie_id, "genres": genres, "index": index}
        )


def render_movie_grid(recs: list[dict], key_prefix: str = "", cols: int = 5):
    """
    Render a row-based grid of movie cards with poster images, similarity
    score bars, genre tags, and "Add to watch history" buttons.
    """
    if not recs:
        st.info("No recommendations found — try adjusting filters or the mode.")
        return

    # Fetch all posters concurrently
    with st.spinner("Loading posters …"):
        with ThreadPoolExecutor(max_workers=5) as ex:
            poster_results = list(
                ex.map(fetch_poster, [r["movie_id"] for r in recs])
            )

    for row_start in range(0, len(recs), cols):
        chunk = recs[row_start: row_start + cols]
        poster_chunk = poster_results[row_start: row_start + cols]
        st_cols = st.columns(len(chunk))

        for col, rec, (poster_url, _) in zip(st_cols, chunk, poster_chunk):
            score_pct = max(0, min(100, int(rec["score"] * 100)))
            genre_str = " · ".join(rec["genres"][:3]) if rec.get("genres") else "—"
            uid = f"{key_prefix}{rec['movie_id']}"

            with col:
                st.markdown('<div class="movie-wrap">', unsafe_allow_html=True)

                # Poster
                if poster_url:
                    st.image(poster_url, use_container_width=True)
                else:
                    st.markdown(
                        '<div style="background:rgba(255,255,255,0.04);border:1px solid '
                        'rgba(255,255,255,0.08);border-radius:8px;height:195px;display:flex;'
                        'align-items:center;justify-content:center;color:#334155;font-size:1.8rem;">'
                        "🎬</div>",
                        unsafe_allow_html=True,
                    )

                # Title
                st.markdown(
                    f'<p class="movie-title">{rec["title"]}</p>',
                    unsafe_allow_html=True,
                )

                # Score bar
                st.markdown(
                    f'<p class="score-label">Match: {score_pct}%</p>'
                    f'<div class="score-track">'
                    f'<div class="score-fill" style="width:{score_pct}%;"></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Genres
                st.markdown(
                    f'<p class="genre-tags">{genre_str}</p>',
                    unsafe_allow_html=True,
                )

                # Watch-history button
                if rec["in_history"]:
                    st.markdown(
                        '<p class="in-history-label">✅ Watched</p>',
                        unsafe_allow_html=True,
                    )
                else:
                    if st.button("+ Watched", key=f"btn_{uid}", use_container_width=True):
                        _add_to_history(
                            rec["title"], rec["movie_id"], rec["genres"], rec["index"]
                        )
                        st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

        if row_start + cols < len(recs):
            st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<h2 style="color:#e2e8f0;font-size:1.1rem;font-weight:700;margin-bottom:0.1rem;">⚙️ Settings</h2>',
        unsafe_allow_html=True,
    )

    # ── Recommendation mode ──────────────────────────────────────────────
    st.markdown(
        '<p style="font-size:0.8rem;color:#64748b;margin:0.9rem 0 0.3rem;">Recommendation Mode</p>',
        unsafe_allow_html=True,
    )

    mode_options = ["Content-Based (TF-IDF)", "Collaborative (NMF)", "Hybrid (Best)"]
    if st.session_state.watch_history:
        mode_options.append("Personalized (For You)")

    if not has_cf:
        selected_label = "Content-Based (TF-IDF)"
        st.info("Run **main.py** to unlock Collaborative, Hybrid & Personalized modes.", icon="ℹ️")
    else:
        selected_label = st.radio(
            "mode_selector",
            mode_options,
            index=2,
            label_visibility="collapsed",
        )

    mode_map = {
        "Content-Based (TF-IDF)": "Content-Based",
        "Collaborative (NMF)":    "Collaborative",
        "Hybrid (Best)":          "Hybrid",
        "Personalized (For You)": "Personalized",
    }
    selected_mode = mode_map[selected_label]

    mode_desc = {
        "Content-Based":  "Finds movies similar to your pick using **TF-IDF** vectors on plot, genres, cast & crew.",
        "Collaborative":  "Discovers movies with **similar latent patterns** via NMF matrix factorization.",
        "Hybrid":         "**60 % TF-IDF** content similarity + **40 % NMF** collaborative filtering.",
        "Personalized":   "Builds a **user preference vector** from your watch history for fully tailored picks.",
    }
    st.caption(mode_desc[selected_mode])

    st.divider()

    # ── Top-N slider ─────────────────────────────────────────────────────
    st.markdown(
        '<p style="font-size:0.8rem;color:#64748b;margin:0 0 0.3rem;">Number of Recommendations</p>',
        unsafe_allow_html=True,
    )
    top_n = st.slider("top_n", min_value=5, max_value=20, value=10, step=5, label_visibility="collapsed")

    st.divider()

    # ── Genre filter ─────────────────────────────────────────────────────
    st.markdown(
        '<p style="font-size:0.8rem;color:#64748b;margin:0 0 0.3rem;">🎭 Genre Filter</p>',
        unsafe_allow_html=True,
    )
    genre_filter = st.multiselect(
        "genre_filter",
        options=all_genres,
        placeholder="All genres",
        label_visibility="collapsed",
    )

    st.divider()

    # ── Watch history ─────────────────────────────────────────────────────
    st.markdown(
        '<p style="font-size:0.8rem;color:#64748b;margin:0 0 0.5rem;">📺 Watch History</p>',
        unsafe_allow_html=True,
    )

    history = st.session_state.watch_history
    if history:
        st.markdown(
            f'<p style="font-size:0.72rem;color:#475569;margin:0 0 0.4rem;">'
            f'{len(history)} movie{"s" if len(history) != 1 else ""} watched</p>',
            unsafe_allow_html=True,
        )
        for idx, h in enumerate(history[-10:]):
            c1, c2 = st.columns([5, 1])
            with c1:
                label = h["title"]
                st.markdown(
                    f'<div class="hist-row"><span>🎬 {label}</span></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("✕", key=f"rm_{idx}", help="Remove"):
                    st.session_state.watch_history = [
                        item for item in history if item["title"] != h["title"]
                    ]
                    st.rerun()

        if len(history) > 10:
            st.caption(f"… and {len(history) - 10} more")

        if st.button("🗑️ Clear All History", use_container_width=True):
            st.session_state.watch_history = []
            st.rerun()

        # Taste profile
        st.divider()
        st.markdown(
            '<p style="font-size:0.8rem;color:#64748b;margin:0 0 0.6rem;">🎯 Your Taste Profile</p>',
            unsafe_allow_html=True,
        )
        genre_counts: dict[str, int] = {}
        for h in history:
            for g in h.get("genres", []):
                genre_counts[g] = genre_counts.get(g, 0) + 1

        if genre_counts:
            top5 = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            max_v = max(c for _, c in top5)
            for g, c in top5:
                bar = int(c / max_v * 100)
                st.markdown(
                    f'<div class="taste-row">'
                    f'<div class="taste-label"><span>{g}</span><span>{c}</span></div>'
                    f'<div class="taste-track">'
                    f'<div class="taste-fill" style="width:{bar}%;"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
    else:
        st.caption("No history yet. Click **+ Watched** on any recommendation card to add movies.")


# ---------------------------------------------------------------------------
# MAIN CONTENT
# ---------------------------------------------------------------------------

# ── App header ───────────────────────────────────────────────────────────
st.markdown(
    '<div class="cinch-header">'
    "<h1>🎬 CineMatch</h1>"
    "<p>AI-powered movie recommendations — Content-Based Filtering · Collaborative Filtering · Hybrid</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Stats row ────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("🎬 Movies", f"{len(movies):,}")
with c2:
    algo_name = {"Content-Based": "TF-IDF", "Collaborative": "NMF CF",
                 "Hybrid": "TF-IDF + NMF", "Personalized": "NMF + History"}
    st.metric("🧠 Algorithm", algo_name.get(selected_mode, selected_mode))
with c3:
    st.metric("📺 Watched", len(history))
with c4:
    st.metric("🎭 Genres", len(all_genres))

st.divider()

# ── Movie search ─────────────────────────────────────────────────────────
movies_list = movies["title"].dropna().drop_duplicates().sort_values().tolist()

search_col, btn_col = st.columns([4, 1])
with search_col:
    selected_movie = st.selectbox(
        "🎥  Search for a movie you like:",
        movies_list,
        index=0,
    )
with btn_col:
    st.markdown("<br>", unsafe_allow_html=True)
    recommend_clicked = st.button("🔍 Recommend", type="primary", use_container_width=True)

# ── Recommendation results ───────────────────────────────────────────────
if recommend_clicked:
    if not selected_movie:
        st.warning("Please select a movie first.")
    else:
        badge_cls = {
            "Content-Based": "badge-cb",
            "Collaborative":  "badge-cf",
            "Hybrid":         "badge-hyb",
            "Personalized":   "badge-pers",
        }.get(selected_mode, "badge-hyb")

        st.markdown(
            f'<div class="section-title">'
            f"<h3>🎯 Recommended — <em>{selected_movie}</em></h3>"
            f'<span class="badge {badge_cls}">{selected_mode}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

        recs = recommend(
            selected_movie,
            selected_mode,
            top_n,
            genre_filter or None,
        )

        if not recs:
            st.warning(
                "No recommendations found. Try clearing the genre filter or switching the mode."
            )
        else:
            render_movie_grid(recs, key_prefix="main_")

# ── Personalized "For You" section ───────────────────────────────────────
if history and has_cf:
    st.markdown(
        '<div class="section-title">'
        "<h3>✨ Personalized For You</h3>"
        '<span style="font-size:0.8rem;color:#475569;margin-left:0.5rem;">Based on your watch history</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    pers_recs = personalized_for_you(top_n=5)
    if pers_recs:
        render_movie_grid(pers_recs, key_prefix="pers_")
    else:
        st.info("Add more movies to your watch history to unlock personalized recommendations.")

# ── Footer ───────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<p class="footer-text">'
    "CineMatch &nbsp;|&nbsp; TF-IDF Content-Based Filtering &nbsp;·&nbsp; "
    "NMF Collaborative Filtering &nbsp;·&nbsp; Hybrid Recommender "
    "&nbsp;|&nbsp; Powered by TMDB"
    "</p>",
    unsafe_allow_html=True,
)