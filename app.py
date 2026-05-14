import os
import re
import sqlite3
import zipfile
import urllib.request
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import bcrypt
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
    "Happy 😊":       {"Comedy", "Family", "Animation"},
    "Tense 🎬":       {"Thriller", "Crime", "Mystery"},
    "Thoughtful 🧠":  {"Drama", "Documentary", "History"},
    "Adventurous 🚀": {"Action", "Adventure", "Sci-Fi"},
    "Romantic 💕":    {"Romance"},
}

GENRE_SENTIMENT_MAP: Dict[str, str] = {
    "Animation": "Positive 😊", "Comedy": "Positive 😊", "Family": "Positive 😊",
    "Fantasy": "Positive 😊", "Adventure": "Positive 😊", "Musical": "Positive 😊",
    "Romance": "Positive 😊", "Horror": "Negative 😟", "Thriller": "Negative 😟",
    "Crime": "Negative 😟", "War": "Negative 😟", "Film-Noir": "Negative 😟",
    "Action": "Neutral 😐", "Drama": "Neutral 😐", "Sci-Fi": "Neutral 😐",
    "Mystery": "Neutral 😐", "Documentary": "Neutral 😐", "History": "Neutral 😐",
    "Western": "Neutral 😐", "IMAX": "Neutral 😐",
}

MOVIE_SYNOPSES = {
    "Toy Story (1995)": "A cowboy doll is profoundly threatened and jealous when a new spaceman action figure supplants him as top toy in a boy's bedroom.",
    "Pulp Fiction (1994)": "The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption.",
    "The Shawshank Redemption (1994)": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
    "Forrest Gump (1994)": "The history of the United States from the 1950s to the 1970s unfolds from the perspective of an Alabama man with an IQ of 75.",
    "Schindler's List (1993)": "In German-occupied Poland during World War II, industrialist Oskar Schindler gradually becomes concerned for his Jewish workforce.",
    "The Silence of the Lambs (1991)": "A young FBI cadet must receive the help of an incarcerated and manipulative cannibal killer to help catch another serial killer.",
    "Jurassic Park (1993)": "A pragmatic paleontologist is tasked with protecting a couple of kids after a power failure causes the park's cloned dinosaurs to run loose.",
    "The Matrix (1999)": "When a beautiful stranger leads computer hacker Neo to a forbidding underworld, he discovers the shocking truth about the life he knows.",
    "Star Wars: Episode IV - A New Hope (1977)": "Luke Skywalker joins forces with a Jedi Knight, a cocky pilot, a Wookiee and two droids to save the galaxy from the Empire.",
    "default": "A compelling film that takes audiences on an unforgettable journey. Widely praised for its storytelling, performances, and direction.",
}

FILTERED_DATA_CSV_LOCAL = os.path.join(DATA_DIR, "filtered_data.csv")
FILTERED_MOVIES_DATA_CSV_LOCAL = os.path.join(DATA_DIR, "filtered_movies_data.csv")
ML_LATEST_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ML_LATEST_SMALL_DIR = os.path.join(DATA_DIR, "ml-latest-small")

# ─────────────────────────────────────────────────────────────────
#  CSS — Dark cinematic theme with DM Sans + Space Mono
# ─────────────────────────────────────────────────────────────────
DARK_CARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.control-panel {
  background: linear-gradient(135deg, #0d1117 0%, #111827 100%);
  border: 1px solid #1f2937; border-radius: 16px;
  padding: 1.6rem 1.8rem 1.2rem; margin-bottom: 1.4rem;
  position: relative; overflow: hidden;
}
.control-panel::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, #6366f1, #06b6d4, #10b981);
  border-radius: 16px 16px 0 0;
}
.control-panel-title {
  font-family: 'Space Mono', monospace; font-size: .85rem; font-weight: 700;
  color: #475569; text-transform: uppercase; letter-spacing: .1em; margin-bottom: 1rem;
}

.movie-detail-card {
  background: #0d1117; border: 1px solid #1f2937; border-radius: 12px;
  padding: 1.2rem 1.4rem; margin-bottom: 1rem;
}
.movie-title-card { font-size: 1.3rem; font-weight: 700; color: #f1f5f9; margin-bottom: .4rem; }
.movie-meta-line { font-size: .88rem; color: #94a3b8; margin-bottom: .22rem; }
.movie-meta-label { font-weight: 600; color: #e2e8f0; }
.movie-rating { color: #fbbf24; font-weight: 700; }
.movie-synopsis {
  font-size: .84rem; color: #6b7280; margin-top: .6rem; line-height: 1.6;
  border-top: 1px solid #1f2937; padding-top: .6rem;
}

.rec-card {
  background: #0d1117; border: 1px solid #1f2937; border-radius: 10px;
  padding: .95rem 1.15rem; margin-bottom: .55rem;
}
.rec-title { font-size: 1rem; font-weight: 700; color: #f1f5f9; margin-bottom: .25rem; }
.rec-meta  { font-size: .81rem; color: #6b7280; }
.rec-score {
  font-family: 'Space Mono', monospace; font-size: .76rem; color: #10b981;
  font-weight: 700; background: #052e16; border-radius: 4px;
  padding: 1px 7px; display: inline-block; margin-top: .3rem;
}
.rec-synopsis { font-size: .81rem; color: #6b7280; margin-top: .45rem; line-height: 1.55; }

.why-tags { display: flex; flex-wrap: wrap; gap: 5px; margin-top: .5rem; }
.why-tag {
  font-size: .73rem; font-weight: 600; border-radius: 4px; padding: 2px 9px;
  display: inline-flex; align-items: center; gap: 3px;
}
.tag-similarity { background: #1e3a5f; color: #60a5fa; }
.tag-mood       { background: #052e16; color: #34d399; }
.tag-rating     { background: #451a03; color: #fb923c; }
.tag-anchor     { background: #1e1b4b; color: #a78bfa; }
.tag-history    { background: #2d1b69; color: #818cf8; }

.filter-banner {
  border-radius: 8px; padding: .6rem 1rem;
  font-size: .84rem; font-weight: 600; margin-bottom: .75rem;
}
.banner-mood    { background: #052e16; border: 1px solid #10b981; color: #34d399; }
.banner-exclude { background: #2d1b69; border: 1px solid #7c3aed; color: #a78bfa; }
.banner-both    { background: #0c1a35; border: 1px solid #3b82f6; color: #60a5fa; }

.section-hdr {
  font-family: 'Space Mono', monospace; font-size: .78rem; font-weight: 700;
  color: #374151; text-transform: uppercase; letter-spacing: .08em;
  margin: 1.3rem 0 .65rem;
}
.section-divider { border: none; border-top: 1px solid #1f2937; margin: 1.3rem 0; }

.helper-msg {
  background: #0c1a35; border: 1px dashed #1e3a5f; border-radius: 10px;
  padding: 1.2rem 1.5rem; color: #3b82f6; font-size: .9rem;
  text-align: center; margin: 1rem 0;
}

.skeleton {
  background: linear-gradient(90deg, #111827 25%, #1f2937 50%, #111827 75%);
  background-size: 200% 100%; animation: shimmer 1.4s infinite;
  border-radius: 6px; margin-bottom: .45rem;
}
.sk-h { height: 17px; } .sk-m { height: 13px; width: 55%; }
.sk-b { height: 12px; } .sk-b2 { height: 12px; width: 72%; }
@keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }

.gauge-val {
  font-family: 'Space Mono', monospace; font-size: 2.1rem;
  font-weight: 700; color: #10b981; text-align: center;
}
.gauge-label {
  font-size: .74rem; color: #475569; text-transform: uppercase;
  letter-spacing: .06em; text-align: center;
}

.guest-banner {
  background: #1c1400; border: 1px solid #92400e; border-radius: 8px;
  padding: .55rem 1rem; color: #fbbf24; font-size: .84rem;
  font-weight: 600; margin-bottom: 1rem;
}
</style>
"""

SKELETON_HTML = """
<div style="background:#0d1117;border:1px solid #1f2937;border-radius:10px;padding:.95rem 1.15rem;margin-bottom:.55rem;">
  <div class="skeleton sk-h" style="width:55%;"></div>
  <div class="skeleton sk-m"></div>
  <div class="skeleton sk-b"></div>
  <div class="skeleton sk-b2"></div>
</div>
"""


# ─────────────────────────────────────────────────────────────────
#  Auth — SQLite with bcrypt, WAL mode for concurrent sessions
# ─────────────────────────────────────────────────────────────────
def _db_connect() -> sqlite3.Connection:
    """Open SQLite with WAL journal mode and a safe timeout."""
    db_path = os.path.join(ROOT_DIR, "users.db")
    conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_auth_db() -> None:
    with _db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def _validate_password_strength(password: str) -> Optional[str]:
    """Return an error string if password is too weak, else None."""
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if not any(c.isupper() for c in password):
        return "Password must contain at least one uppercase letter."
    if not any(c.isdigit() for c in password):
        return "Password must contain at least one number."
    return None


def _signup(username: str, password: str, confirm: str) -> tuple:
    if len(username.strip()) < 3:
        return False, "Username must be at least 3 characters."
    if password != confirm:
        return False, "Passwords do not match."
    err = _validate_password_strength(password)
    if err:
        return False, err
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        with _db_connect() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username.strip().lower(), hashed),
            )
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "Username already taken — please choose another."


def _login(username: str, password: str) -> tuple:
    with _db_connect() as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE username = ?",
            (username.strip().lower(),),
        ).fetchone()
    if not row:
        return False, "Username not found."
    if bcrypt.checkpw(password.encode(), row[0].encode()):
        return True, "Login successful."
    return False, "Incorrect password."


def render_auth_screen() -> bool:
    """Returns True if user is logged in. Shows login/signup/guest UI otherwise."""
    _init_auth_db()

    if st.session_state.get("logged_in"):
        return True

    # Centered auth container CSS
    st.markdown("""
    <style>
    .auth-container {
        max-width: 520px;
        margin: 60px auto;
        padding: 2rem 2rem 1.5rem;
        background: #0d1117;
        border: 1px solid #1f2937;
        border-radius: 18px;
        box-shadow: 0 0 30px rgba(0,0,0,0.35);
    }

    .auth-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        color: #f8fafc;
    }

    div[data-baseweb="tab-list"] {
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # Start centered container
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)

    st.markdown(
        "<div class='auth-title'>🎬 RecoMind</div>",
        unsafe_allow_html=True
    )

    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    with tab_login:
        uname = st.text_input("Username", key="login_user")
        pwd = st.text_input("Password", type="password", key="login_pwd")

        col_login, col_guest = st.columns(2)

        with col_login:
            if st.button("Login", use_container_width=True):
                ok, msg = _login(uname, pwd)

                if ok:
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = uname.strip().lower()
                    st.rerun()
                else:
                    st.error(msg)

        with col_guest:
            if st.button("👤 Guest / Demo", use_container_width=True):
                st.session_state["logged_in"] = True
                st.session_state["username"] = "guest"
                st.rerun()

    with tab_signup:
        new_user = st.text_input("Choose Username", key="signup_user")

        new_pwd = st.text_input(
            "Choose Password (min 8 chars, 1 uppercase, 1 number)",
            type="password",
            key="signup_pwd"
        )

        confirm_pwd = st.text_input(
            "Confirm Password",
            type="password",
            key="signup_confirm"
        )

        if st.button("Create Account", use_container_width=True):
            ok, msg = _signup(new_user, new_pwd, confirm_pwd)

            if ok:
                st.success(msg + " Please log in.")
            else:
                st.error(msg)

    st.markdown("</div>", unsafe_allow_html=True)

    return False

# ─────────────────────────────────────────────────────────────────
#  Data layer
# ─────────────────────────────────────────────────────────────────
def _ensure_data_downloaded() -> Tuple[str, str]:
    if os.path.exists(MOVIES_CSV_LOCAL) and os.path.exists(RATINGS_CSV_LOCAL):
        return MOVIES_CSV_LOCAL, RATINGS_CSV_LOCAL
    os.makedirs(DATA_DIR, exist_ok=True)
    em = os.path.join(ML_LATEST_SMALL_DIR, "movies.csv")
    er = os.path.join(ML_LATEST_SMALL_DIR, "ratings.csv")
    if os.path.exists(em) and os.path.exists(er):
        return em, er
    zp = os.path.join(DATA_DIR, "ml-latest-small.zip")
    with st.spinner("Downloading MovieLens dataset…"):
        urllib.request.urlretrieve(ML_LATEST_SMALL_URL, zp)
    with zipfile.ZipFile(zp, "r") as zf:
        zf.extractall(DATA_DIR)
    if not (os.path.exists(em) and os.path.exists(er)):
        raise FileNotFoundError("Download complete but CSVs not found.")
    return em, er


@st.cache_data(show_spinner=False)
def load_and_prepare_data(top_n_movies: int, top_n_users: int):
    fs = None
    if os.path.exists(FILTERED_MOVIES_DATA_CSV_LOCAL):
        fs = FILTERED_MOVIES_DATA_CSV_LOCAL
    elif os.path.exists(FILTERED_DATA_CSV_LOCAL):
        fs = FILTERED_DATA_CSV_LOCAL
    if fs is not None:
        df = pd.read_csv(fs)
        if not {"userId", "title", "rating"}.issubset(df.columns):
            raise ValueError("Filtered CSV missing required columns.")
        gmap = (df[["title", "genres"]].drop_duplicates("title")
                .set_index("title")["genres"].to_dict()) if "genres" in df.columns else {}
        umm = df.pivot_table(index="userId", columns="title",
                             values="rating", aggfunc="mean").fillna(0)
        return df, df[["title"]].drop_duplicates(), umm, gmap

    mp, rp = _ensure_data_downloaded()
    movies  = pd.read_csv(mp)
    ratings = pd.read_csv(rp)
    df = pd.merge(ratings, movies, on="movieId")
    df.dropna(inplace=True)
    df = df[df["title"].isin(df["title"].value_counts().head(top_n_movies).index)]
    df = df[df["userId"].isin(df["userId"].value_counts().head(top_n_users).index)]
    gmap = movies[["title", "genres"]].drop_duplicates("title").set_index("title")["genres"].to_dict()
    umm  = df.pivot_table(index="userId", columns="title",
                          values="rating", aggfunc="mean").fillna(0)
    return df, movies, umm, gmap


@st.cache_data(show_spinner=False)
def compute_avg_ratings(top_n_movies: int, top_n_users: int) -> Dict[str, float]:
    df, _, _, _ = load_and_prepare_data(top_n_movies, top_n_users)
    return df.groupby("title")["rating"].mean().round(2).to_dict()


@st.cache_resource(show_spinner=True)
def build_recommender(top_n_movies: int, top_n_users: int):
    _, _, umm, gmap = load_and_prepare_data(top_n_movies, top_n_users)
    X     = umm.to_numpy(dtype=np.float32)
    norms = np.linalg.norm(X, axis=0)
    norms[norms == 0] = 1e-8
    Xn    = X / norms
    sim   = Xn.T @ Xn
    titles = umm.columns
    t2i    = {t: i for i, t in enumerate(titles)}
    return sim, gmap, titles, t2i


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────
def extract_year(title: str) -> str:
    m = re.search(r"\((\d{4})\)\s*$", str(title))
    return m.group(1) if m else "—"

def clean_title(full: str) -> str:
    return re.sub(r"\s*\(\d{4}\)\s*$", "", str(full)).strip()

def genres_set(gs: str) -> Set[str]:
    if not gs or not str(gs).strip():
        return set()
    return {g.strip() for g in str(gs).split("|") if g.strip()}

def collect_genre_options(gmap: Dict[str, str]) -> List[str]:
    found: Set[str] = set()
    for g in gmap.values():
        found.update(genres_set(g))
    return sorted(found)

def genre_sentiment_label(gs: str) -> str:
    if not gs or gs == "—": return "Neutral 😐"
    s = {GENRE_SENTIMENT_MAP.get(t, "Neutral 😐") for t in genres_set(gs)}
    if "Positive 😊" in s: return "Positive 😊"
    if "Negative 😟" in s: return "Negative 😟"
    return "Neutral 😐"

def excluded_match(gs: str, excl: FrozenSet[str]) -> bool:
    return bool(excl and genres_set(gs) & set(excl))


# ─────────────────────────────────────────────────────────────────
#  TMDB
# ─────────────────────────────────────────────────────────────────
def _tmdb_key() -> Optional[str]:
    k = os.environ.get("TMDB_API_KEY", "").strip()
    if k: return k
    try:    return str(st.secrets["TMDB_API_KEY"]).strip() or None
    except: return None


@st.cache_data(ttl=86400, show_spinner=False)
def _tmdb_result(api_key: str, ct: str, yr: str) -> Optional[dict]:
    if not api_key or not ct: return None
    params: Dict[str, str] = {"api_key": api_key, "query": ct}
    if yr and yr.isdigit(): params["year"] = yr
    try:
        r = requests.get("https://api.themoviedb.org/3/search/movie",
                         params=params, timeout=12)
        if r.status_code != 200: return None
        res = (r.json() or {}).get("results") or []
        return res[0] if res else None
    except requests.RequestException:
        return None

def get_poster(api_key: Optional[str], full: str) -> Optional[str]:
    if not api_key: return None
    y  = extract_year(full)
    ct = clean_title(full)
    res = _tmdb_result(api_key, ct, y if y != "—" else "")
    if not res: return None
    p = res.get("poster_path")
    return f"https://image.tmdb.org/t/p/w342{p}" if p else None

def get_synopsis(api_key: Optional[str], full: str) -> str:
    y  = extract_year(full)
    ct = clean_title(full)
    if api_key:
        res = _tmdb_result(api_key, ct, y if y != "—" else "")
        if res:
            ov = (res.get("overview") or "").strip()
            if ov: return ov
    return MOVIE_SYNOPSES.get(full, MOVIE_SYNOPSES["default"])

def show_poster(api_key: Optional[str], full: str) -> None:
    if not api_key: return
    pu = get_poster(api_key, full)
    if pu: st.image(pu, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
#  Recommendation engine
# ─────────────────────────────────────────────────────────────────
def recommend(
    sim_matrix: np.ndarray,
    movie_titles: pd.Index,
    gmap: Dict[str, str],
    t2i: Dict[str, int],
    movie_name: str,
    top_k: int,
    avg_ratings: Dict[str, float],
    excl: FrozenSet[str] = frozenset(),
    mood_genres: Set[str] = set(),
    mood_boost: float = 0.15,
    max_scan: int = 4000,
) -> pd.DataFrame:
    if movie_name not in t2i:
        return pd.DataFrame(columns=["movie", "score", "genres", "mood_match"])

    idx        = t2i[movie_name]
    sim_scores = sim_matrix[idx].copy()

    if mood_genres:
        for i, t in enumerate(movie_titles):
            if i != idx and genres_set(gmap.get(t, "")) & mood_genres:
                sim_scores[i] = min(1.0, sim_scores[i] + mood_boost)

    max_r = max(avg_ratings.values()) if avg_ratings else 5.0
    if max_r == 0: max_r = 5.0

    hybrid = np.array([
        sim_scores[i] * 0.7 + (avg_ratings.get(str(movie_titles[i]), 0.0) / max_r) * 0.3
        for i in range(len(movie_titles))
    ], dtype=np.float32)

    picked: List[int] = []
    scanned = 0
    for i in np.argsort(-hybrid).tolist():
        if i == idx: continue
        scanned += 1
        if scanned > max_scan: break
        if excluded_match(gmap.get(movie_titles[i], ""), excl): continue
        picked.append(i)
        if len(picked) >= top_k: break

    if not picked:
        return pd.DataFrame(columns=["movie", "score", "genres", "mood_match"])

    df = pd.DataFrame({"movie": movie_titles[picked], "score": hybrid[picked]})
    df["genres"]     = df["movie"].map(lambda t: gmap.get(t, ""))
    df["mood_match"] = df["genres"].apply(
        lambda g: bool(genres_set(g) & mood_genres) if mood_genres else False)
    return df


# ─────────────────────────────────────────────────────────────────
#  Explanation tags
# ─────────────────────────────────────────────────────────────────
def build_why_tags(
    title: str, score: float, mood_match: bool,
    avg_ratings: Dict[str, float], anchor: str, from_history: bool = False,
) -> str:
    tags = []

    if score > 0.80:
        tags.append('<span class="why-tag tag-similarity">🔥 Very strong match</span>')
    elif score > 0.55:
        tags.append('<span class="why-tag tag-similarity">👥 Similar viewers liked this</span>')
    elif score > 0.35:
        tags.append('<span class="why-tag tag-similarity">🔍 Worth exploring</span>')

    if mood_match:
        tags.append('<span class="why-tag tag-mood">🎭 Matches your mood</span>')
    r = avg_ratings.get(title, 0.0)
    if r >= 4.0:
        tags.append(f'<span class="why-tag tag-rating">⭐ Highly rated ({r})</span>')
    if anchor:
        tags.append(
            f'<span class="why-tag tag-anchor">🎬 Fans of {clean_title(anchor)[:22]} also watched</span>'
        )
    if from_history:
        tags.append('<span class="why-tag tag-history">🕓 From watch history</span>')
    if not tags:
        tags.append('<span class="why-tag tag-similarity">👥 Collaborative match</span>')
    return '<div class="why-tags">' + "".join(tags) + "</div>"


# ─────────────────────────────────────────────────────────────────
#  Render helpers
# ─────────────────────────────────────────────────────────────────
def render_movie_card(
    full_title: str, genres_str: str,
    avg_rating: Optional[float], api_key: Optional[str],
) -> None:
    year = extract_year(full_title)
    ct   = clean_title(full_title)
    gd   = genres_str.replace("|", "  ·  ") if genres_str and genres_str != "—" else "—"
    rd   = f"⭐ {avg_rating}" if avg_rating else "—"
    syn  = get_synopsis(api_key, full_title)

    pc, ic = st.columns([1, 2.8])
    with pc:
        pu = get_poster(api_key, full_title) if api_key else None
        if pu:
            st.image(pu, use_container_width=True)
        else:
            st.markdown(
                "<div style='background:#0d1117;border:1px solid #1f2937;border-radius:8px;"
                "height:200px;display:flex;align-items:center;justify-content:center;"
                "color:#374151;font-size:2.5rem;'>🎬</div>",
                unsafe_allow_html=True,
            )
    with ic:
        st.markdown(
            f"<div class='movie-detail-card'>"
            f"<div class='movie-title-card'>{ct} <span style='color:#374151'>({year})</span></div>"
            f"<div class='movie-meta-line'><span class='movie-meta-label'>Genres</span>&nbsp;—&nbsp;{gd}</div>"
            f"<div class='movie-meta-line'><span class='movie-meta-label'>Rating</span>&nbsp;—&nbsp;"
            f"<span class='movie-rating'>{rd}</span></div>"
            f"<div class='movie-synopsis'>{syn}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_rec_cards(
    recs: pd.DataFrame,
    avg_ratings: Dict[str, float],
    api_key: Optional[str],
    top_k: int,
    anchor: str = "",
    from_history: bool = False,
) -> None:
    if recs.empty:
        st.warning("No recommendations found for the current filters.")
        return

    st.markdown("<div class='section-hdr'>🎯 Recommendations</div>", unsafe_allow_html=True)

    for _, row in recs.iterrows():
        title  = str(row["movie"])
        score  = float(row["score"])
        graw   = str(row.get("genres", ""))
        gd     = graw.replace("|", "  ·  ") if graw else "—"
        year   = extract_year(title)
        ct     = clean_title(title)
        rating = avg_ratings.get(title)
        rstr   = f"⭐ {rating}" if rating else "—"
        mmatch = bool(row.get("mood_match", False))
        syn    = get_synopsis(api_key, title)
        why    = build_why_tags(title, score, mmatch, avg_ratings, anchor, from_history)

        pc, ic = st.columns([1, 3])
        with pc:
            show_poster(api_key, title)
        with ic:
            st.markdown(
                f"<div class='rec-card'>"
                f"<div class='rec-title'>{ct} <span style='color:#374151'>({year})</span></div>"
                f"<div class='rec-meta'>{gd}&nbsp;·&nbsp;{rstr}</div>"
                f"<span class='rec-score'>score {score:.4f}</span>"
                f"<div class='rec-synopsis'>{syn}</div>"
                f"{why}"
                f"</div>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────
#  Sentiment analysis section
# ─────────────────────────────────────────────────────────────────
def render_sentiment_section(selected_movie: str) -> None:
    import plotly.express as px
    from textblob import TextBlob

    st.markdown("<div class='section-hdr'>🎭 Audience Sentiment Analysis</div>", unsafe_allow_html=True)
    st.caption(
        "NLP sentiment scoring via TextBlob. "
        "Currently using simulated prototype reviews for offline demonstration — "
        "future scope: live integration via review APIs or web scraping."
    )

    reviews = SAMPLE_REVIEWS.get(selected_movie, SAMPLE_REVIEWS["default"])

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
    fig = px.pie(
        sc, names="Sentiment", values="Count", color="Sentiment", hole=0.38,
        color_discrete_map={
            "Positive 😊": "#10b981", "Neutral 😐": "#475569", "Negative 😟": "#ef4444"
        },
        title="Sentiment Distribution",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                      font_family="DM Sans", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    avg_pol = scored_df["Polarity"].mean()
    overall = "Positive 😊" if avg_pol > 0.05 else ("Negative 😟" if avg_pol < -0.05 else "Neutral 😐")
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Audience Mood",  overall)
    c2.metric("Avg Polarity Score",     f"{avg_pol:.2f} / 1.0")
    c3.metric("Reviews Analysed",       len(reviews))

    if avg_pol > 0:
        st.success("Audience reaction is mostly positive.")
    elif avg_pol < 0:
        st.error("Audience reaction is mostly negative.")
    else:
        st.info("Audience sentiment is neutral.")

    st.markdown("**✍️ Analyse Your Own Review**")
    user_review = st.text_area(
        "Type a review:", placeholder="e.g. This movie was incredible!", height=90,
        key="user_review_input",
    )
    if user_review.strip():
        up = TextBlob(user_review).sentiment.polarity
        ul = "Positive 😊" if up > 0.05 else ("Negative 😟" if up < -0.05 else "Neutral 😐")
        st.success(f"**Your review sentiment:** {ul}  (Polarity: {up:.3f})")


# ─────────────────────────────────────────────────────────────────
#  Tab 1 — Unified Recommend
# ─────────────────────────────────────────────────────────────────
def render_unified_tab(
    sim_matrix, movie_titles, gmap, t2i,
    avg_ratings, genre_options, tmdb_key, top_k, excluded_frozen,
) -> None:
    # Guest mode banner
    if st.session_state.get("username") == "guest":
        st.markdown(
            "<div class='guest-banner'>"
            "👤 Guest mode — watch history and personalisation won't persist across sessions."
            "</div>",
            unsafe_allow_html=True,
        )

    last = st.session_state["last_built_params"]

    # ── Trending strip ──────────────────────────────────────────
    df_ratings, _, _, _ = load_and_prepare_data(last["top_n_movies"], last["top_n_users"])
    top_counts     = df_ratings["title"].value_counts().head(8)
    trending_4     = [t for t in top_counts.index
                      if not excluded_match(gmap.get(t, ""), excluded_frozen)][:4]

    if trending_4:
        st.markdown("<div class='section-hdr'>🔥 Trending Now</div>", unsafe_allow_html=True)
        tcols = st.columns(len(trending_4))
        for i, t in enumerate(trending_4):
            with tcols[i]:
                r = avg_ratings.get(t, 0)
                st.markdown(
                    f"<div style='background:#0d1117;border:1px solid #1f2937;border-radius:8px;"
                    f"padding:.6rem .8rem;'>"
                    f"<div style='font-size:.84rem;font-weight:700;color:#e2e8f0;"
                    f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>"
                    f"{clean_title(t)}</div>"
                    f"<div style='font-size:.74rem;color:#475569;'>"
                    f"{extract_year(t)}&nbsp;·&nbsp;⭐&nbsp;{r}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Unified Control Panel ────────────────────────────────────
    st.markdown(
        "<div class='control-panel'>"
        "<div class='control-panel-title'>🤖 RecoMind — Unified Engine</div>",
        unsafe_allow_html=True,
    )

    all_sorted = sorted(movie_titles.tolist())
    if excluded_frozen:
        all_sorted = [t for t in all_sorted if not excluded_match(gmap.get(t, ""), excluded_frozen)]

    col_movie, col_genre = st.columns([3, 1])
    with col_genre:
        genre_filter = st.selectbox("🎭 Genre", ["All"] + genre_options, index=0, key="uni_genre")
    if genre_filter != "All":
        all_sorted = [t for t in all_sorted if genre_filter in genres_set(gmap.get(t, ""))]
    with col_movie:
        selected_movie = st.selectbox(
            "🔍 Search & Select Movie",
            options=all_sorted, index=None,
            placeholder="Type to search…", key="uni_movie",
        )

    # Mood row
    mood_options = list(MOOD_GENRE_MAP.keys())
    if "selected_mood" not in st.session_state:
        st.session_state["selected_mood"] = None

    st.markdown(
        "<div style='font-size:.81rem;color:#475569;margin:.55rem 0 .25rem;'>"
        "🎭 Mood <span style='color:#1f2937'>(optional — click to activate)</span></div>",
        unsafe_allow_html=True,
    )
    mcols = st.columns(len(mood_options))
    for i, mood in enumerate(mood_options):
        with mcols[i]:
            active = st.session_state["selected_mood"] == mood
            if st.button(f"{'✓ ' if active else ''}{mood}", key=f"mood_{i}", use_container_width=True):
                st.session_state["selected_mood"] = None if active else mood

    st.markdown("</div>", unsafe_allow_html=True)

    selected_mood = st.session_state["selected_mood"]
    mood_genres   = MOOD_GENRE_MAP.get(selected_mood, set()) if selected_mood else set()

    # Active filter banners
    has_mood = selected_mood is not None
    has_excl = bool(excluded_frozen)
    if has_mood and has_excl:
        st.markdown(
            f"<div class='filter-banner banner-both'>"
            f"🎭 {selected_mood}&nbsp;&nbsp;+&nbsp;&nbsp;"
            f"🚫 Excluding: {', '.join(sorted(excluded_frozen))}</div>",
            unsafe_allow_html=True,
        )
    elif has_mood:
        st.markdown(
            f"<div class='filter-banner banner-mood'>"
            f"🎭 Mood active: {selected_mood} — boosting {', '.join(sorted(mood_genres))}</div>",
            unsafe_allow_html=True,
        )
    elif has_excl:
        st.markdown(
            f"<div class='filter-banner banner-exclude'>"
            f"🚫 Wellbeing filter: excluding {', '.join(sorted(excluded_frozen))}</div>",
            unsafe_allow_html=True,
        )

    # Empty state
    if not selected_movie and not has_mood:
        st.markdown(
            "<div class='helper-msg'>"
            "Select a movie above — or pick a mood for instant recommendations.</div>",
            unsafe_allow_html=True,
        )
        return

    # Auto-anchor on mood only
    anchor = selected_movie
    if not anchor and has_mood:
        mood_titles = [
            t for t in movie_titles
            if genres_set(gmap.get(t, "")) & mood_genres
            and not excluded_match(gmap.get(t, ""), excluded_frozen)
        ]
        if not mood_titles:
            st.warning("No movies found for this mood with the current exclusions.")
            return
        anchor = max(mood_titles, key=lambda t: avg_ratings.get(t, 0.0))
        st.info(f"🎭 No movie selected — showing top picks for **{selected_mood}** mood")
        st.caption(f"Auto-selected anchor: **{clean_title(anchor)}** (top-rated in mood genre)")

    # Movie detail card
    if selected_movie:
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        render_movie_card(
            selected_movie,
            gmap.get(selected_movie, "—") or "—",
            avg_ratings.get(selected_movie),
            tmdb_key,
        )

    # Skeleton → load → render recs
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    sk = st.empty()
    sk.markdown("".join([SKELETON_HTML] * 3), unsafe_allow_html=True)

    recs = recommend(
        sim_matrix=sim_matrix, movie_titles=movie_titles, gmap=gmap, t2i=t2i,
        movie_name=anchor, top_k=top_k, avg_ratings=avg_ratings,
        excl=excluded_frozen, mood_genres=mood_genres,
    )

    # Update watch history (skip for guest — ephemeral session only)
    hist: List[str] = st.session_state.get("watch_history", [])
    if anchor and anchor in t2i:
        hist = [m for m in hist if m != anchor]
        hist.insert(0, anchor)
        st.session_state["watch_history"] = hist[:5]

    sk.empty()

    if excluded_frozen:
        st.info(f"🚫 Wellbeing filter active — {', '.join(sorted(excluded_frozen))} excluded from all results.")

    st.caption(
        "**Hybrid ranking:** 70% collaborative similarity + 30% audience rating. "
        "Score ≈ 1.0 = strongest match."
    )
    render_rec_cards(recs, avg_ratings, tmdb_key, top_k, anchor=anchor or "")

    # Sentiment section (only when a movie is explicitly selected)
    if selected_movie:
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        render_sentiment_section(selected_movie)

    # Watch-history personalisation
    hist = st.session_state.get("watch_history", [])
    if len(hist) > 1:
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-hdr'>🕓 Personalised — Based on Watch History</div>",
            unsafe_allow_html=True,
        )
        st.caption("  ·  ".join(clean_title(h) for h in hist))
        all_recs: List[pd.DataFrame] = []
        for ht in hist:
            if ht in t2i:
                hr = recommend(sim_matrix, movie_titles, gmap, t2i, ht, 20,
                               avg_ratings, excluded_frozen)
                all_recs.append(hr)
        if all_recs:
            combined = pd.concat(all_recs).drop_duplicates("movie")
            combined = combined[~combined["movie"].isin(hist)]
            combined = (combined.groupby("movie", as_index=False)
                        .agg({"score": "mean", "genres": "first", "mood_match": "any"})
                        .sort_values("score", ascending=False).head(8).reset_index(drop=True))
            render_rec_cards(combined, avg_ratings, tmdb_key, 8, from_history=True)


# ─────────────────────────────────────────────────────────────────
#  Tab 2 — Dashboard
# ─────────────────────────────────────────────────────────────────
def render_dashboard_tab(gmap: Dict[str, str], avg_ratings: Dict[str, float], last: dict) -> None:
    import plotly.express as px

    df_dash, _, _, _ = load_and_prepare_data(last["top_n_movies"], last["top_n_users"])

    total_movies  = df_dash["title"].nunique()
    total_ratings = len(df_dash)
    overall_avg   = df_dash["rating"].mean()
    most_genre_movie = (max(gmap, key=lambda t: len(genres_set(gmap[t])), default="—")
                        if gmap else "—")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🎬 Movies in Subset",  total_movies)
    k2.metric("📊 Total Ratings",     f"{total_ratings:,}")
    k3.metric("⭐ Avg Rating",        f"{overall_avg:.2f}")
    k4.metric("🎭 Most Diverse Movie", clean_title(most_genre_movie)[:20] if gmap else "—")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    st.markdown("#### ⭐ Rating Distribution")
    rc = df_dash["rating"].value_counts().sort_index().reset_index()
    rc.columns = ["Rating", "Count"]
    fig1 = px.bar(rc, x="Rating", y="Count", color="Count",
                  color_continuous_scale="teal", title="How users rated movies")
    fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#e2e8f0", font_family="DM Sans")
    st.plotly_chart(fig1, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🎬 Most Rated Movies")
        tm = df_dash["title"].value_counts().head(10).reset_index()
        tm.columns = ["Movie", "Ratings"]
        tm["Movie"] = tm["Movie"].map(clean_title)
        fig2 = px.bar(tm, x="Ratings", y="Movie", orientation="h",
                      color="Ratings", color_continuous_scale="Blues")
        fig2.update_layout(yaxis={"categoryorder": "total ascending"},
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0", font_family="DM Sans", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown("#### 🎭 Genre Breakdown")
        if gmap:
            all_g: List[str] = []
            for g in gmap.values():
                all_g.extend(genres_set(g))
            gs = pd.Series(all_g).value_counts().reset_index()
            gs.columns = ["Genre", "Count"]
            fig3 = px.pie(gs, names="Genre", values="Count", hole=0.38)
            fig3.update_traces(textposition="inside", textinfo="percent+label")
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                               font_family="DM Sans", showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    st.markdown("#### 🧠 Emotional Tone Distribution")
    if gmap:
        sl = [genre_sentiment_label(g) for g in gmap.values()]
        sd = pd.Series(sl).value_counts().reset_index()
        sd.columns = ["Sentiment", "Count"]
        fig4 = px.bar(sd, x="Sentiment", y="Count", color="Sentiment",
                      color_discrete_map={
                          "Positive 😊": "#10b981",
                          "Neutral 😐":  "#475569",
                          "Negative 😟": "#ef4444",
                      }, title="Content emotional distribution across dataset")
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0", font_family="DM Sans", showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### 🏆 Top Rated Movies (≥ 50 ratings)")
    count_map = df_dash["title"].value_counts().to_dict()
    top_rated = {t: r for t, r in avg_ratings.items() if count_map.get(t, 0) >= 50}
    if top_rated:
        tr_df = (pd.DataFrame(list(top_rated.items()), columns=["Title", "Avg Rating"])
                 .sort_values("Avg Rating", ascending=False).head(10))
        tr_df["Title"] = tr_df["Title"].map(clean_title)
        fig5 = px.bar(tr_df, x="Avg Rating", y="Title", orientation="h",
                      color="Avg Rating", color_continuous_scale="RdYlGn",
                      range_color=[3.0, 5.0])
        fig5.update_layout(yaxis={"categoryorder": "total ascending"},
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0", font_family="DM Sans", showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("#### 📡 Dataset Quality Snapshot")
    gc1, gc2, gc3 = st.columns(3)
    pct_positive = (
        sum(1 for g in gmap.values() if genre_sentiment_label(g) == "Positive 😊")
        / max(len(gmap), 1) * 100
    ) if gmap else 0
    unique_genres = len({g for gs in gmap.values() for g in genres_set(gs)}) if gmap else 0

    for col, val, label in [
        (gc1, f"{overall_avg:.2f}", "Avg Rating"),
        (gc2, f"{pct_positive:.0f}%", "Positive-tone Movies"),
        (gc3, str(unique_genres), "Unique Genres"),
    ]:
        with col:
            st.markdown(
                f"<div class='gauge-val'>{val}</div>"
                f"<div class='gauge-label'>{label}</div>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="RecoMind", page_icon="🎬", layout="wide")
    st.markdown(DARK_CARD_CSS, unsafe_allow_html=True)

    if not render_auth_screen():
        st.stop()

    st.title("🎬 RecoMind")
    st.caption(
        "Collaborative filtering &nbsp;·&nbsp; Hybrid ranking &nbsp;·&nbsp; "
        "Mood-aware &nbsp;·&nbsp; Wellbeing filter &nbsp;·&nbsp; TextBlob NLP"
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        with st.form("train_form"):
            top_n_movies = st.slider("Top Movies (training)", 50, 300, 100, 10)
            top_n_users  = st.slider("Active Users (training)", 100, 2000, 500, 100)
            build_submit = st.form_submit_button("🔄 Rebuild Model")
        top_k = st.slider("Results to show", 5, 30, 10, 1)
        st.divider()
        st.subheader("🚫 Wellbeing Filter")
        st.caption("Excludes genres globally across all results.")

    if "watch_history" not in st.session_state:
        st.session_state["watch_history"] = []
    if "last_built_params" not in st.session_state or build_submit:
        st.session_state["last_built_params"] = {
            "top_n_movies": top_n_movies, "top_n_users": top_n_users
        }

    last = st.session_state["last_built_params"]

    with st.spinner("⚙️ Building recommendation model…"):
        sim_matrix, gmap, movie_titles, t2i = build_recommender(
            last["top_n_movies"], last["top_n_users"]
        )

    avg_ratings   = compute_avg_ratings(last["top_n_movies"], last["top_n_users"])
    genre_options = collect_genre_options(gmap)
    tmdb_key      = _tmdb_key()

    with st.sidebar:
        global_excluded = st.multiselect(
            "Exclude genres", options=genre_options, default=[], key="global_excluded"
        ) if genre_options else []
        excluded_frozen = frozenset(global_excluded)
        st.divider()
        st.subheader("🖼️ TMDB Posters")
        if tmdb_key:
            st.success("API key found — posters & synopses enabled.")
        else:
            st.warning("Add `TMDB_API_KEY` to secrets to enable posters.")
        st.divider()
        uname_display = st.session_state.get("username", "")
        label = f"👤 Guest session" if uname_display == "guest" else f"👤 Logged in as **{uname_display}**"
        st.caption(label)
        if st.button("Logout", use_container_width=True):
            st.session_state["logged_in"] = False
            st.session_state["username"]  = ""
            st.session_state["watch_history"] = []
            st.rerun()

    tab_rec, tab_dash = st.tabs(["🎬 Recommend", "📊 Dashboard"])

    with tab_rec:
        render_unified_tab(
            sim_matrix, movie_titles, gmap, t2i,
            avg_ratings, genre_options, tmdb_key, top_k, excluded_frozen,
        )
        st.markdown("---")
        st.caption("RecoMind · Built by Smit Patel 🚀 · MovieLens · Streamlit")

    with tab_dash:
        render_dashboard_tab(gmap, avg_ratings, last)


if __name__ == "__main__":
    main()
