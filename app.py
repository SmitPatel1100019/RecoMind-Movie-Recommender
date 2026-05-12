import os
import re
import zipfile
import urllib.request
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "Data", "MovieLens")

MOVIES_CSV_LOCAL = os.path.join(DATA_DIR, "movies.csv")
RATINGS_CSV_LOCAL = os.path.join(DATA_DIR, "ratings.csv")

SAMPLE_REVIEWS = {
    "default": [
        "This movie was absolutely fantastic, I loved every moment of it!",
        "It was okay, nothing special but not terrible either.",
        "Disappointing experience, the plot made no sense at all.",
    ],
    "Toy Story (1995)": [
        "A timeless classic that appeals to both kids and adults!",
        "Good animation for its time, story is decent.",
        "Felt a bit slow in the middle but overall enjoyable.",
    ],
    "Pulp Fiction (1994)": [
        "Tarantino at his absolute best, a masterpiece of cinema.",
        "Interesting structure but not for everyone.",
        "Too violent and confusing for my taste.",
    ],
    "The Shawshank Redemption (1994)": [
        "One of the greatest films ever made, deeply moving.",
        "A solid drama with great performances.",
        "Overrated in my opinion, too slow paced.",
    ],
    "Forrest Gump (1994)": [
        "Beautiful, emotional and uplifting from start to finish!",
        "Tom Hanks is great but the story drags at times.",
        "Not my kind of movie but I can see why people like it.",
    ],
}

MOOD_GENRE_MAP = {
    "Happy 😊":        {"Comedy", "Family", "Animation"},
    "Tense 🎬":        {"Thriller", "Crime", "Mystery"},
    "Thoughtful 🧠":   {"Drama", "Documentary", "History"},
    "Adventurous 🚀":  {"Action", "Adventure", "Sci-Fi"},
    "Romantic 💕":     {"Romance"},
}

GENRE_SENTIMENT_MAP: Dict[str, str] = {
    "Animation":   "Positive 😊",
    "Comedy":      "Positive 😊",
    "Family":      "Positive 😊",
    "Fantasy":     "Positive 😊",
    "Adventure":   "Positive 😊",
    "Musical":     "Positive 😊",
    "Romance":     "Positive 😊",
    "Horror":      "Negative 😟",
    "Thriller":    "Negative 😟",
    "Crime":       "Negative 😟",
    "War":         "Negative 😟",
    "Film-Noir":   "Negative 😟",
    "Action":      "Neutral 😐",
    "Drama":       "Neutral 😐",
    "Sci-Fi":      "Neutral 😐",
    "Mystery":     "Neutral 😐",
    "Documentary": "Neutral 😐",
    "History":     "Neutral 😐",
    "Western":     "Neutral 😐",
    "IMAX":        "Neutral 😐",
}

# Fallback synopses when TMDB key is not set or movie not found
MOVIE_SYNOPSES = {
    "Toy Story (1995)": "A cowboy doll is profoundly threatened and jealous when a new spaceman action figure supplants him as top toy in a boy's bedroom.",
    "Pulp Fiction (1994)": "The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption.",
    "The Shawshank Redemption (1994)": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
    "Forrest Gump (1994)": "The history of the United States from the 1950s to the 1970s unfolds from the perspective of an Alabama man with an IQ of 75.",
    "Schindler's List (1993)": "In German-occupied Poland during World War II, industrialist Oskar Schindler gradually becomes concerned for his Jewish workforce after witnessing their persecution by the Nazis.",
    "The Silence of the Lambs (1991)": "A young FBI cadet must receive the help of an incarcerated and manipulative cannibal killer to help catch another serial killer.",
    "Jurassic Park (1993)": "A pragmatic paleontologist visiting an almost-complete amusement park is tasked with protecting a couple of kids after a power failure causes the park's cloned dinosaurs to run loose.",
    "The Matrix (1999)": "When a beautiful stranger leads computer hacker Neo to a forbidding underworld, he discovers the shocking truth — the life he knows is the elaborate deception of an evil cyber-intelligence.",
    "Star Wars: Episode IV - A New Hope (1977)": "Luke Skywalker joins forces with a Jedi Knight, a cocky pilot, a Wookiee and two droids to save the galaxy from the Empire's world-destroying battle station.",
    "default": "A compelling film that takes audiences on an unforgettable journey. Widely praised for its storytelling, performances, and direction.",
}

FILTERED_DATA_CSV_LOCAL = os.path.join(DATA_DIR, "filtered_data.csv")
FILTERED_MOVIES_DATA_CSV_LOCAL = os.path.join(DATA_DIR, "filtered_movies_data.csv")

ML_LATEST_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ML_LATEST_SMALL_DIR = os.path.join(DATA_DIR, "ml-latest-small")

DARK_CARD_CSS = """
<style>
.search-card{background:#0e1117;border:1px solid #2d2d2d;border-radius:10px;
  padding:1.4rem 1.6rem 1rem;margin-bottom:1.2rem}
.search-card-title{font-size:1.25rem;font-weight:700;color:#fff;margin-bottom:.2rem}
.search-card-sub{font-size:.82rem;color:#888;margin-bottom:1rem}
.found-count{font-size:.9rem;font-weight:600;color:#ccc;margin:.5rem 0 .8rem}

.movie-detail-card{background:#0e1117;border:1px solid #2d2d2d;border-radius:10px;
  padding:1.2rem 1.4rem;margin-bottom:1rem}
.movie-title-card{font-size:1.3rem;font-weight:700;color:#fff;margin-bottom:.5rem}
.movie-meta-line{font-size:.9rem;color:#ccc;margin-bottom:.3rem}
.movie-meta-label{font-weight:600;color:#fff}
.movie-rating{color:#f5c518;font-weight:700}
.movie-synopsis{font-size:.88rem;color:#aaa;margin-top:.6rem;line-height:1.55;
  border-top:1px solid #2d2d2d;padding-top:.6rem}

.rec-card{background:#0e1117;border:1px solid #2d2d2d;border-radius:8px;
  padding:.9rem 1.1rem;margin-bottom:.5rem}
.rec-title{font-size:1rem;font-weight:700;color:#fff}
.rec-meta{font-size:.82rem;color:#999;margin-top:.2rem}
.rec-score{font-size:.85rem;color:#02C39A;font-weight:600}
.mood-badge{display:inline-block;background:#1a3a2a;color:#02C39A;border-radius:4px;
  padding:1px 8px;font-size:.78rem;margin-left:6px}

.filter-panel{background:#111827;border:1px solid #1e3a5f;border-radius:12px;
  padding:1.4rem 1.6rem;margin-bottom:1.4rem}
.filter-panel-title{font-size:1.1rem;font-weight:700;color:#60a5fa;
  margin-bottom:.25rem;letter-spacing:.02em}
.filter-panel-sub{font-size:.8rem;color:#6b7280;margin-bottom:1rem}

.wellbeing-panel{background:#111827;border:1px solid #3d1f5e;border-radius:12px;
  padding:1.4rem 1.6rem;margin-bottom:1.4rem}
.wellbeing-panel-title{font-size:1.1rem;font-weight:700;color:#a78bfa;
  margin-bottom:.25rem;letter-spacing:.02em}
.wellbeing-panel-sub{font-size:.8rem;color:#6b7280;margin-bottom:1rem}

.helper-msg{background:#0f1f2e;border:1px dashed #1e3a5f;border-radius:8px;
  padding:.9rem 1.2rem;color:#4b7fa8;font-size:.9rem;text-align:center;margin:1rem 0}
.active-banner{border-radius:8px;padding:.7rem 1.1rem;
  font-size:.88rem;font-weight:600;margin-bottom:.8rem}
.mood-active{background:#0d2d1a;border:1px solid #02C39A;color:#02C39A}
.well-active{background:#1e0d3a;border:1px solid #a78bfa;color:#a78bfa}
.both-active{background:#0d1f30;border:1px solid #60a5fa;color:#60a5fa}

.section-divider{border:none;border-top:1px solid #1f2937;margin:1.6rem 0}
.rec-section-header{font-size:1.05rem;font-weight:700;color:#e5e7eb;
  margin:.5rem 0 .9rem;letter-spacing:.01em}
</style>
"""


# ─────────────────────────────────────────────────────────────────
#  Data helpers
# ─────────────────────────────────────────────────────────────────
def _ensure_data_downloaded() -> Tuple[str, str]:
    if os.path.exists(MOVIES_CSV_LOCAL) and os.path.exists(RATINGS_CSV_LOCAL):
        return MOVIES_CSV_LOCAL, RATINGS_CSV_LOCAL
    os.makedirs(DATA_DIR, exist_ok=True)
    extracted_movies  = os.path.join(ML_LATEST_SMALL_DIR, "movies.csv")
    extracted_ratings = os.path.join(ML_LATEST_SMALL_DIR, "ratings.csv")
    if os.path.exists(extracted_movies) and os.path.exists(extracted_ratings):
        return extracted_movies, extracted_ratings
    zip_path = os.path.join(DATA_DIR, "ml-latest-small.zip")
    with st.spinner("Downloading MovieLens …"):
        urllib.request.urlretrieve(ML_LATEST_SMALL_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    if not (os.path.exists(extracted_movies) and os.path.exists(extracted_ratings)):
        raise FileNotFoundError("MovieLens download completed, but CSVs not found.")
    return extracted_movies, extracted_ratings


@st.cache_data(show_spinner=False)
def load_and_prepare_data(top_n_movies: int, top_n_users: int):
    filtered_source = None
    if os.path.exists(FILTERED_MOVIES_DATA_CSV_LOCAL):
        filtered_source = FILTERED_MOVIES_DATA_CSV_LOCAL
    elif os.path.exists(FILTERED_DATA_CSV_LOCAL):
        filtered_source = FILTERED_DATA_CSV_LOCAL

    if filtered_source is not None:
        df = pd.read_csv(filtered_source)
        if not {"userId", "title", "rating"}.issubset(df.columns):
            raise ValueError("Filtered CSV missing required columns.")
        genres_map = (
            df[["title", "genres"]].drop_duplicates("title")
            .set_index("title")["genres"].to_dict()
        ) if "genres" in df.columns else {}
        user_movie_matrix = (
            df.pivot_table(index="userId", columns="title",
                           values="rating", aggfunc="mean").fillna(0)
        )
        movies = df[["title"]].drop_duplicates()
        return df, movies, user_movie_matrix, genres_map

    movies_path, ratings_path = _ensure_data_downloaded()
    movies  = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    df = pd.merge(ratings, movies, on="movieId")
    df.dropna(inplace=True)
    top_movies   = df["title"].value_counts().head(top_n_movies).index
    df           = df[df["title"].isin(top_movies)]
    active_users = df["userId"].value_counts().head(top_n_users).index
    df           = df[df["userId"].isin(active_users)]
    genres_map   = (
        movies[["title", "genres"]].drop_duplicates("title")
        .set_index("title")["genres"].to_dict()
    )
    user_movie_matrix = (
        df.pivot_table(index="userId", columns="title",
                       values="rating", aggfunc="mean").fillna(0)
    )
    return df, movies, user_movie_matrix, genres_map


@st.cache_data(show_spinner=False)
def compute_avg_ratings(top_n_movies: int, top_n_users: int) -> Dict[str, float]:
    df, _, _, _ = load_and_prepare_data(top_n_movies, top_n_users)
    return df.groupby("title")["rating"].mean().round(1).to_dict()


@st.cache_resource(show_spinner=True)
def build_recommender(top_n_movies: int, top_n_users: int):
    _, _, user_movie_matrix, genres_map = load_and_prepare_data(top_n_movies, top_n_users)
    X          = user_movie_matrix.to_numpy(dtype=np.float32)
    col_norms  = np.linalg.norm(X, axis=0)
    col_norms[col_norms == 0] = 1e-8
    Xn         = X / col_norms
    sim_matrix = Xn.T @ Xn
    movie_titles   = user_movie_matrix.columns
    title_to_index = {t: i for i, t in enumerate(movie_titles)}
    return sim_matrix, genres_map, movie_titles, title_to_index


# ─────────────────────────────────────────────────────────────────
#  Utility helpers
# ─────────────────────────────────────────────────────────────────
def extract_year_from_title(title: str) -> str:
    m = re.search(r"\((\d{4})\)\s*$", str(title))
    return m.group(1) if m else "—"

def clean_title_for_tmdb(full_title: str) -> str:
    return re.sub(r"\s*\(\d{4}\)\s*$", "", str(full_title)).strip()

def genres_string_to_set(genres_str: str) -> Set[str]:
    if not genres_str or not str(genres_str).strip():
        return set()
    return {g.strip() for g in str(genres_str).split("|") if g.strip()}

def collect_genre_options(genres_map: Dict[str, str]) -> List[str]:
    found: Set[str] = set()
    for g in genres_map.values():
        found.update(genres_string_to_set(g))
    return sorted(found)

def genre_sentiment_label(genres_str: str) -> str:
    if not genres_str or genres_str == "—":
        return "Neutral 😐"
    tags = genres_string_to_set(genres_str)
    sentiments = {GENRE_SENTIMENT_MAP.get(tag, "Neutral 😐") for tag in tags}
    if "Positive 😊" in sentiments:
        return "Positive 😊"
    if "Negative 😟" in sentiments:
        return "Negative 😟"
    return "Neutral 😐"

def movie_matches_excluded_genres(genres_str: str, excluded: FrozenSet[str]) -> bool:
    if not excluded:
        return False
    return bool(genres_string_to_set(genres_str) & set(excluded))

def _tmdb_api_key() -> Optional[str]:
    k = os.environ.get("TMDB_API_KEY", "").strip()
    if k:
        return k
    try:
        return str(st.secrets["TMDB_API_KEY"]).strip() or None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────
#  TMDB — single cached search call powers BOTH poster + synopsis
# ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def tmdb_search_result(api_key: str, clean_title: str, year_str: str) -> Optional[dict]:
    """One TMDB search → returns first result dict (poster_path + overview inside)."""
    if not api_key or not clean_title:
        return None
    params: Dict[str, str] = {"api_key": api_key, "query": clean_title}
    if year_str and year_str.isdigit():
        params["year"] = year_str
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params=params, timeout=12,
        )
        if r.status_code != 200:
            return None
        results = (r.json() or {}).get("results") or []
        return results[0] if results else None
    except requests.RequestException:
        return None


def tmdb_poster_url(api_key: str, clean_title: str, year_str: str) -> Optional[str]:
    result = tmdb_search_result(api_key, clean_title, year_str)
    if not result:
        return None
    path = result.get("poster_path")
    return f"https://image.tmdb.org/t/p/w342{path}" if path else None


def tmdb_synopsis(api_key: str, clean_title: str, year_str: str) -> Optional[str]:
    """Return movie overview from TMDB — same cached call as poster, no extra request."""
    result = tmdb_search_result(api_key, clean_title, year_str)
    if not result:
        return None
    overview = (result.get("overview") or "").strip()
    return overview if overview else None


def get_poster_url(api_key: Optional[str], full_title: str) -> Optional[str]:
    if not api_key:
        return None
    y = extract_year_from_title(full_title)
    return tmdb_poster_url(api_key, clean_title_for_tmdb(full_title), y if y != "—" else "")

def show_poster_for_title(api_key: Optional[str], full_title: str,
                           caption: Optional[str] = None) -> None:
    if not api_key:
        return
    pu = get_poster_url(api_key, full_title)
    if pu:
        st.image(pu, caption=caption or full_title[:80], use_container_width=True)


# ─────────────────────────────────────────────────────────────────
#  Core CF logic
# ─────────────────────────────────────────────────────────────────
def recommend_from_similarity(
    sim_matrix: np.ndarray,
    movie_titles: pd.Index,
    genres_map: Dict[str, str],
    title_to_index: Dict[str, int],
    movie_name: str,
    top_k: int,
    excluded_genres: FrozenSet[str] = frozenset(),
    mood_genres: Set[str] = set(),
    mood_boost: float = 0.15,
    max_scan: int = 4000,
) -> pd.DataFrame:
    if movie_name not in movie_titles:
        return pd.DataFrame([{
            "movie": movie_name, "score": 0.0,
            "genres": genres_map.get(movie_name, ""), "mood_match": False,
        }])
    movie_index = title_to_index[movie_name]
    scores = sim_matrix[movie_index].copy()

    if mood_genres:
        for i, title in enumerate(movie_titles):
            if i == movie_index:
                continue
            if genres_string_to_set(genres_map.get(title, "")) & mood_genres:
                scores[i] = min(1.0, scores[i] + mood_boost)

    order = np.argsort(-scores)
    picked_idx: List[int] = []
    scanned = 0
    for i in order.tolist():
        if i == movie_index:
            continue
        scanned += 1
        if scanned > max_scan:
            break
        if movie_matches_excluded_genres(genres_map.get(movie_titles[i], ""), excluded_genres):
            continue
        picked_idx.append(i)
        if len(picked_idx) >= top_k:
            break

    if not picked_idx:
        return pd.DataFrame(columns=["movie", "score", "genres", "mood_match"])

    rec_df = pd.DataFrame({
        "movie": movie_titles[picked_idx],
        "score": scores[picked_idx],
    })
    rec_df["genres"]     = rec_df["movie"].map(lambda t: genres_map.get(t, ""))
    rec_df["mood_match"] = rec_df["genres"].apply(
        lambda g: bool(genres_string_to_set(g) & mood_genres) if mood_genres else False
    )
    return rec_df


# ─────────────────────────────────────────────────────────────────
#  Render helpers
# ─────────────────────────────────────────────────────────────────
def render_recommendation_cards(
    recs: pd.DataFrame,
    mood_pick: str,
    avg_ratings: Dict[str, float],
    api_key: Optional[str],
    top_k: int,
) -> None:
    if recs.empty:
        st.warning("No recommendations found. Try adjusting filters or picking a different movie.")
        return

    n_poster = min(5, len(recs))
    if api_key and n_poster:
        st.markdown("<div class='rec-section-header'>🖼️ Top Picks — Posters</div>",
                    unsafe_allow_html=True)
        cols = st.columns(n_poster)
        for j in range(n_poster):
            t = str(recs["movie"].iloc[j])
            with cols[j]:
                show_poster_for_title(api_key, t, t[:40])
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    st.markdown("<div class='rec-section-header'>📋 All Recommendations</div>",
                unsafe_allow_html=True)
    for _, row in recs.iterrows():
        title        = str(row["movie"])
        score        = float(row["score"])
        genres_raw   = str(row.get("genres", ""))
        genres_disp  = genres_raw.replace("|", ", ") if genres_raw and genres_raw != "—" else "—"
        year         = extract_year_from_title(title)
        clean        = clean_title_for_tmdb(title)
        rating       = avg_ratings.get(title)
        rating_str   = f"⭐ {rating}" if rating else "—"
        is_mood      = bool(row.get("mood_match", False))
        mood_badge   = "<span class='mood-badge'>🎭 Mood match</span>" if is_mood else ""
        sentiment    = genre_sentiment_label(genres_raw)

        st.markdown(
            f"<div class='rec-card'>"
            f"<div class='rec-title'>{clean} ({year}){mood_badge}</div>"
            f"<div class='rec-meta'>🎭 {genres_disp} &nbsp;|&nbsp; "
            f"<span class='movie-rating'>{rating_str}</span> &nbsp;|&nbsp; "
            f"{sentiment} &nbsp;|&nbsp; "
            f"<span class='rec-score'>Similarity: {score:.4f}</span></div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_movie_detail_card(
    full_title: str,
    genres_str: str,
    avg_rating: Optional[float],
    api_key: Optional[str],
) -> None:
    year           = extract_year_from_title(full_title)
    clean          = clean_title_for_tmdb(full_title)
    genres_display = genres_str.replace("|", ", ") if genres_str and genres_str != "—" else "—"
    rating_display = f"⭐ {avg_rating}" if avg_rating else "—"

    # ── Synopsis: TMDB first, hardcoded fallback, generic last ────
    synopsis: Optional[str] = None
    if api_key:
        y = extract_year_from_title(full_title)
        synopsis = tmdb_synopsis(api_key, clean, y if y != "—" else "")
    if not synopsis:
        synopsis = MOVIE_SYNOPSES.get(full_title, MOVIE_SYNOPSES["default"])

    poster_col, info_col = st.columns([1, 2.8])
    with poster_col:
        pu = get_poster_url(api_key, full_title) if api_key else None
        if pu:
            st.image(pu, use_container_width=True)
        else:
            st.markdown(
                "<div style='background:#1a1a2e;border-radius:8px;height:200px;"
                "display:flex;align-items:center;justify-content:center;"
                "color:#555;font-size:2rem;'>🎬</div>",
                unsafe_allow_html=True,
            )
    with info_col:
        st.markdown(
            f"<div class='movie-detail-card'>"
            f"<div class='movie-title-card'>{clean} ({year})</div>"
            f"<div class='movie-meta-line'><span class='movie-meta-label'>Genres:</span> {genres_display}</div>"
            f"<div class='movie-meta-line'><span class='movie-meta-label'>Rating:</span> "
            f"<span class='movie-rating'>{rating_display}</span></div>"
            f"<div class='movie-synopsis'><b>Synopsis:</b> {synopsis}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────
#  Mood Mode tab
# ─────────────────────────────────────────────────────────────────
def render_mood_tab(
    sim_matrix, movie_titles, genres_map, title_to_index,
    avg_ratings, excluded_frozen, tmdb_key, top_k,
) -> None:
    st.markdown(
        "<div class='filter-panel'>"
        "<div class='filter-panel-title'>🎭 Mood Mode</div>"
        "<div class='filter-panel-sub'>"
        "Pick your current mood — recommendations update instantly, no button needed."
        "</div></div>",
        unsafe_allow_html=True,
    )

    all_titles = sorted(movie_titles.tolist())
    if excluded_frozen:
        all_titles = [
            t for t in all_titles
            if not movie_matches_excluded_genres(genres_map.get(t, ""), excluded_frozen)
        ]

    anchor_movie = st.selectbox(
        "📽️ Start from a movie you like",
        options=all_titles, index=None,
        placeholder="Search or select a movie…",
        key="mood_anchor",
    )

    mood_options = list(MOOD_GENRE_MAP.keys())
    cols_mood    = st.columns(len(mood_options))
    if "selected_mood" not in st.session_state:
        st.session_state["selected_mood"] = None

    for i, mood in enumerate(mood_options):
        with cols_mood[i]:
            is_active = st.session_state["selected_mood"] == mood
            label     = f"{'✅ ' if is_active else ''}{mood}"
            if st.button(label, key=f"mood_btn_{i}", use_container_width=True):
                st.session_state["selected_mood"] = None if st.session_state["selected_mood"] == mood else mood

    selected_mood = st.session_state["selected_mood"]
    mood_genres   = MOOD_GENRE_MAP.get(selected_mood, set()) if selected_mood else set()

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    has_mood     = selected_mood is not None
    has_excluded = bool(excluded_frozen)

    if has_mood and has_excluded:
        st.markdown(
            f"<div class='active-banner both-active'>"
            f"🎭 Mood: <b>{selected_mood}</b> &nbsp;+&nbsp; "
            f"🚫 Excluding: <b>{', '.join(sorted(excluded_frozen))}</b> — both filters active."
            f"</div>", unsafe_allow_html=True,
        )
    elif has_mood:
        st.markdown(
            f"<div class='active-banner mood-active'>"
            f"🎭 Mood: <b>{selected_mood}</b> — boosting genres: {', '.join(sorted(mood_genres))}"
            f"</div>", unsafe_allow_html=True,
        )
    elif has_excluded:
        st.markdown(
            f"<div class='active-banner well-active'>"
            f"🚫 Wellbeing filter active — excluding: <b>{', '.join(sorted(excluded_frozen))}</b>"
            f"</div>", unsafe_allow_html=True,
        )

    if not anchor_movie and not has_mood:
        st.markdown(
            "<div class='helper-msg'>"
            "💡 Select a mood above or pick a starting movie to see personalised recommendations."
            "</div>", unsafe_allow_html=True,
        )
        return

    if not anchor_movie and has_mood:
        mood_titles = [
            t for t in movie_titles
            if genres_string_to_set(genres_map.get(t, "")) & mood_genres
            and not movie_matches_excluded_genres(genres_map.get(t, ""), excluded_frozen)
        ]
        if not mood_titles:
            st.warning("No movies found for this mood with the current exclusions.")
            return
        anchor_movie = max(mood_titles, key=lambda t: avg_ratings.get(t, 0.0))
        st.caption(f"🔍 Auto-selected anchor: **{clean_title_for_tmdb(anchor_movie)}** (top-rated in mood genre)")

    with st.spinner("Finding mood-matched movies…"):
        recs = recommend_from_similarity(
            sim_matrix=sim_matrix, movie_titles=movie_titles,
            genres_map=genres_map, title_to_index=title_to_index,
            movie_name=anchor_movie, top_k=top_k,
            excluded_genres=excluded_frozen, mood_genres=mood_genres,
        )

    if recs.empty:
        st.warning("No results with these filters. Try a different mood or fewer exclusions.")
        return

    render_recommendation_cards(recs, selected_mood or "No preference", avg_ratings, tmdb_key, top_k)


# ─────────────────────────────────────────────────────────────────
#  Wellbeing Filter tab
# ─────────────────────────────────────────────────────────────────
def render_wellbeing_tab(
    sim_matrix, movie_titles, genres_map, title_to_index,
    avg_ratings, genre_options, tmdb_key, top_k,
) -> None:
    st.markdown(
        "<div class='wellbeing-panel'>"
        "<div class='wellbeing-panel-title'>🛡️ Wellbeing Filter</div>"
        "<div class='wellbeing-panel-sub'>"
        "Exclude genres you'd rather avoid — recommendations refresh automatically."
        "</div></div>",
        unsafe_allow_html=True,
    )

    wb_excluded = st.multiselect(
        "🚫 Genres to exclude", options=genre_options, default=[],
        placeholder="Choose genres to block…", key="wb_excluded",
        help="Results update instantly as you add or remove genres.",
    )
    wb_frozen = frozenset(wb_excluded)

    all_titles_wb = sorted(movie_titles.tolist())
    if wb_frozen:
        all_titles_wb = [
            t for t in all_titles_wb
            if not movie_matches_excluded_genres(genres_map.get(t, ""), wb_frozen)
        ]

    anchor_wb = st.selectbox(
        "📽️ Pick a movie to base recommendations on",
        options=all_titles_wb, index=None,
        placeholder="Search or select…", key="wb_anchor",
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    if wb_excluded:
        st.markdown(
            f"<div class='active-banner well-active'>"
            f"🚫 Blocking: {', '.join(sorted(wb_excluded))} — "
            f"these genres are completely removed from results."
            f"</div>", unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='helper-msg'>"
            "💡 No genres excluded yet. Select genres above to filter them out."
            "</div>", unsafe_allow_html=True,
        )

    if not anchor_wb:
        st.markdown(
            "<div class='helper-msg'>"
            "👆 Select a starting movie above to see filtered recommendations."
            "</div>", unsafe_allow_html=True,
        )
        return

    with st.spinner("Applying wellbeing filter and generating recommendations…"):
        recs_wb = recommend_from_similarity(
            sim_matrix=sim_matrix, movie_titles=movie_titles,
            genres_map=genres_map, title_to_index=title_to_index,
            movie_name=anchor_wb, top_k=top_k,
            excluded_genres=wb_frozen, mood_genres=set(),
        )

    if recs_wb.empty:
        st.warning("No recommendations left after exclusions. Try removing some blocked genres.")
        return

    if wb_excluded:
        st.caption(
            f"✅ Showing **{len(recs_wb)}** clean recommendations — "
            f"all free of: {', '.join(sorted(wb_excluded))}."
        )

    render_recommendation_cards(recs_wb, "No preference", avg_ratings, tmdb_key, top_k)


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="RecoMind", page_icon="🎬", layout="wide")
    st.markdown(DARK_CARD_CSS, unsafe_allow_html=True)
    st.title("🎬 RecoMind — Movie Recommendations")
    st.caption("Collaborative filtering · Cosine similarity · TextBlob NLP")

    with st.sidebar:
        st.header("⚙️ Settings")
        with st.form("train_form"):
            top_n_movies = st.slider("Top Movies (training filter)", 50, 300, 100, 10)
            top_n_users  = st.slider("Active Users (training filter)", 100, 2000, 500, 100)
            build_submit = st.form_submit_button("Build/Refresh model")
        top_k = st.slider("Recommendations to show", 5, 30, 10, 1)

    if "watch_history" not in st.session_state:
        st.session_state["watch_history"] = []
    if "last_built_params" not in st.session_state or build_submit:
        st.session_state["last_built_params"] = {
            "top_n_movies": top_n_movies, "top_n_users": top_n_users
        }

    last = st.session_state.get("last_built_params", {"top_n_movies": 100, "top_n_users": 500})

    with st.spinner("Building model…"):
        sim_matrix, genres_map, movie_titles, title_to_index = build_recommender(
            top_n_movies=last["top_n_movies"], top_n_users=last["top_n_users"],
        )

    avg_ratings   = compute_avg_ratings(last["top_n_movies"], last["top_n_users"])
    genre_options = collect_genre_options(genres_map)
    tmdb_key      = _tmdb_api_key()

    with st.sidebar:
        st.divider()
        st.subheader("🚫 Global Wellbeing Filter")
        st.caption("Applies to Search & Recommendations tab.")
        global_excluded = st.multiselect(
            "Exclude genres globally", options=genre_options, default=[],
            key="global_excluded",
        ) if genre_options else []
        excluded_frozen = frozenset(global_excluded)

        st.divider()
        st.subheader("🖼️ TMDB Posters")
        if tmdb_key:
            st.success("API key found — posters + synopses enabled.")
        else:
            st.warning("Set `TMDB_API_KEY` in secrets for posters and live synopses.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎬 Search & Recommendations",
        "🎭 Mood Mode",
        "🛡️ Wellbeing Filter",
        "📊 Dashboard",
    ])

    # ── Tab 1 ─────────────────────────────────────────────────────
    with tab1:
        import plotly.express as px

        df_ratings, _, _, _ = load_and_prepare_data(last["top_n_movies"], last["top_n_users"])
        top_counts = df_ratings["title"].value_counts().head(10)
        trending = pd.DataFrame({
            "Movie":             top_counts.index,
            "Ratings in subset": top_counts.values.astype(int),
        })
        trending["Year"]   = trending["Movie"].map(extract_year_from_title)
        trending["Genres"] = trending["Movie"].map(lambda t: genres_map.get(t, "—"))
        if excluded_frozen:
            trending = trending[trending["Genres"].map(
                lambda gs: not movie_matches_excluded_genres(str(gs), excluded_frozen)
            )]

        st.subheader("🔥 Trending in this Dataset")
        if trending.empty:
            st.info("No trending titles with the current genre exclusions.")
        else:
            st.table(trending)

        st.divider()
        st.markdown(
            "<div class='search-card'>"
            "<div class='search-card-title'>🔍 Search &amp; 🎯 Recommendations</div>"
            "<div class='search-card-sub'>Filter movies by attributes, then generate recommendations</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        col_name, col_genre = st.columns([2, 1])
        with col_name:
            search_q = st.text_input(
                "Search by Movie Name", value="",
                placeholder="e.g. Toy Story, Matrix, Spider-Man…",
            )
        with col_genre:
            genre_filter = st.selectbox(
                "Filter by Genre", options=["All genres"] + genre_options, index=0,
            )

        all_sorted = sorted(movie_titles.tolist())
        if excluded_frozen:
            all_sorted = [
                t for t in all_sorted
                if not movie_matches_excluded_genres(genres_map.get(t, ""), excluded_frozen)
            ]
        if genre_filter != "All genres":
            all_sorted = [
                t for t in all_sorted
                if genre_filter in genres_string_to_set(genres_map.get(t, ""))
            ]

        q = search_q.strip().lower()
        filtered_titles = [t for t in all_sorted if q in t.lower()] if q else all_sorted
        if not filtered_titles:
            st.warning("No matches — reset search or adjust genre filters.")
            filtered_titles = all_sorted

        st.markdown(
            f"<div class='found-count'>Found <b>{len(filtered_titles)}</b> "
            f"movie{'s' if len(filtered_titles) != 1 else ''} matching your criteria.</div>",
            unsafe_allow_html=True,
        )

        selected_movie = st.selectbox(
            "Pick a movie to analyse", options=filtered_titles,
            index=None, placeholder="Select a movie…",
        )

        if selected_movie:
            st.markdown("---")
            render_movie_detail_card(
                selected_movie,
                genres_map.get(selected_movie, "—") or "—",
                avg_ratings.get(selected_movie),
                tmdb_key,
            )
        else:
            st.info("Search and select a movie above, then press **Recommend**.")

        if selected_movie:
            st.divider()
            st.subheader("🎭 Audience Sentiment Analysis")
            st.caption("Sentiment analysis of audience reviews using TextBlob NLP.")
            from textblob import TextBlob

            reviews = SAMPLE_REVIEWS.get(selected_movie, SAMPLE_REVIEWS["default"])
            if selected_movie not in SAMPLE_REVIEWS:
                st.warning("⚠️ No specific reviews for this title — generic placeholders used.")

            scored = []
            for r in reviews:
                pol   = TextBlob(r).sentiment.polarity
                label = "Positive 😊" if pol > 0.05 else ("Negative 😟" if pol < -0.05 else "Neutral 😐")
                scored.append({"Review": r, "Sentiment": label, "Polarity": round(pol, 3)})

            scored_df = pd.DataFrame(scored)
            st.markdown("**📋 Sample Reviews**")
            st.table(scored_df)

            sc = scored_df["Sentiment"].value_counts().reset_index()
            sc.columns = ["Sentiment", "Count"]
            fig_sent = px.pie(
                sc, names="Sentiment", values="Count", color="Sentiment",
                color_discrete_map={
                    "Positive 😊": "#2ecc71", "Neutral 😐": "#95a5a6", "Negative 😟": "#e74c3c"
                },
                title="Sentiment Distribution", hole=0.3,
            )
            fig_sent.update_traces(textposition="inside", textinfo="percent+label")
            fig_sent.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#ffffff", showlegend=False
            )
            st.plotly_chart(fig_sent, use_container_width=True)

            avg_pol = scored_df["Polarity"].mean()
            overall = "Positive 😊" if avg_pol > 0.05 else ("Negative 😟" if avg_pol < -0.05 else "Neutral 😐")
            c1, c2, c3 = st.columns(3)
            c1.metric("Overall Audience Mood", overall)
            c2.metric("Avg Polarity Score", f"{avg_pol:.2f} / 1.0")
            c3.metric("Total Reviews Analysed", len(reviews))

            st.markdown("**✍️ Analyse Your Own Review**")
            user_review = st.text_area(
                "Type a review:", placeholder="e.g. This movie was incredible!", height=100
            )
            if user_review.strip():
                up = TextBlob(user_review).sentiment.polarity
                ul = "Positive 😊" if up > 0.05 else ("Negative 😟" if up < -0.05 else "Neutral 😐")
                st.success(f"**Your review sentiment:** {ul} (Polarity: {up:.3f})")

        st.divider()
        st.subheader("🎯 Top Recommendations")
        if st.button("Recommend", type="primary"):
            if selected_movie:
                with st.spinner("Generating recommendations…"):
                    recs = recommend_from_similarity(
                        sim_matrix=sim_matrix, movie_titles=movie_titles,
                        movie_name=selected_movie, genres_map=genres_map,
                        title_to_index=title_to_index, top_k=top_k,
                        excluded_genres=excluded_frozen, mood_genres=set(),
                    )
                history: List[str] = st.session_state["watch_history"]
                if selected_movie not in history:
                    history.insert(0, selected_movie)
                    st.session_state["watch_history"] = history[:5]
                if excluded_frozen:
                    st.warning(f"🚫 Excluded: {', '.join(sorted(excluded_frozen))}")
                st.caption(
                    f"Movies most similar to **{selected_movie}** by cosine similarity "
                    "over user rating patterns. Score closer to 1 = more alike."
                )
                render_recommendation_cards(recs, "No preference", avg_ratings, tmdb_key, top_k)
            else:
                st.warning("Please select a movie first.")

        history = st.session_state.get("watch_history", [])
        if history:
            st.divider()
            st.subheader("🕓 Your Recent Picks")
            st.write(" · ".join(history))
            st.subheader("🎯 Based on Your Recent Picks")
            all_recs: List[pd.DataFrame] = []
            for h_title in history:
                if h_title in title_to_index:
                    h_recs = recommend_from_similarity(
                        sim_matrix=sim_matrix, movie_titles=movie_titles,
                        movie_name=h_title, genres_map=genres_map,
                        title_to_index=title_to_index, top_k=20,
                        excluded_genres=excluded_frozen, mood_genres=set(),
                    )
                    all_recs.append(h_recs)
            if all_recs:
                combined = pd.concat(all_recs)
                combined = combined[~combined["movie"].isin(history)]
                combined = (
                    combined.groupby("movie", as_index=False)
                    .agg({"score": "mean", "genres": "first", "mood_match": "any"})
                    .sort_values("score", ascending=False)
                    .head(10).reset_index(drop=True)
                )
                render_recommendation_cards(combined, "No preference", avg_ratings, tmdb_key, 10)

        st.markdown("---")
        st.write("RecoMind | Built by Smit Patel 🚀")

    # ── Tab 2 ─────────────────────────────────────────────────────
    with tab2:
        render_mood_tab(
            sim_matrix=sim_matrix, movie_titles=movie_titles,
            genres_map=genres_map, title_to_index=title_to_index,
            avg_ratings=avg_ratings, excluded_frozen=excluded_frozen,
            tmdb_key=tmdb_key, top_k=top_k,
        )

    # ── Tab 3 ─────────────────────────────────────────────────────
    with tab3:
        render_wellbeing_tab(
            sim_matrix=sim_matrix, movie_titles=movie_titles,
            genres_map=genres_map, title_to_index=title_to_index,
            avg_ratings=avg_ratings, genre_options=genre_options,
            tmdb_key=tmdb_key, top_k=top_k,
        )

    # ── Tab 4 ─────────────────────────────────────────────────────
    with tab4:
        import plotly.express as px

        st.subheader("📊 Dataset Dashboard")
        st.caption("Analysis of the MovieLens training subset.")
        df_dash, _, _, _ = load_and_prepare_data(last["top_n_movies"], last["top_n_users"])

        st.markdown("#### ⭐ Rating Distribution")
        rc = df_dash["rating"].value_counts().sort_index().reset_index()
        rc.columns = ["Rating", "Count"]
        fig1 = px.bar(rc, x="Rating", y="Count", color="Count",
                      color_continuous_scale="Blues", title="How users rated movies")
        fig1.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#ffffff"
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("#### 🎬 Top 10 Most Rated Movies")
        tm = df_dash["title"].value_counts().head(10).reset_index()
        tm.columns = ["Movie", "Number of Ratings"]
        fig2 = px.bar(tm, x="Number of Ratings", y="Movie", orientation="h",
                      color="Number of Ratings", color_continuous_scale="Teal",
                      title="Most rated movies in training subset")
        fig2.update_layout(
            yaxis={"categoryorder": "total ascending"},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#ffffff",
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### 🎭 Genre Distribution")
        if genres_map:
            all_g: List[str] = []
            for g in genres_map.values():
                all_g.extend(genres_string_to_set(g))
            gs = pd.Series(all_g).value_counts().reset_index()
            gs.columns = ["Genre", "Count"]
            fig3 = px.pie(gs, names="Genre", values="Count", title="Genre breakdown", hole=0.3)
            fig3.update_traces(textposition="inside", textinfo="percent+label")
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#ffffff", showlegend=False
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No genre data available.")

        st.markdown("#### 🧠 Sentiment Distribution Across Genres")
        if genres_map:
            sl = [genre_sentiment_label(g) for g in genres_map.values()]
            sd = pd.Series(sl).value_counts().reset_index()
            sd.columns = ["Sentiment", "Count"]
            fig4 = px.bar(sd, x="Sentiment", y="Count", color="Sentiment",
                          color_discrete_map={
                              "Positive 😊": "#2ecc71",
                              "Neutral 😐":  "#95a5a6",
                              "Negative 😟": "#e74c3c",
                          },
                          title="Emotional tone of movies in dataset")
            fig4.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#ffffff"
            )
            st.plotly_chart(fig4, use_container_width=True)


if __name__ == "__main__":
    main()
