import os 
import re
import sqlite3
import zipfile
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Set, Tuple

import bcrypt
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
import requests
import streamlit as st

st.set_page_config(page_title="RecoMind", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "Data", "MovieLens")

MOVIES_CSV_LOCAL = os.path.join(DATA_DIR, "movies.csv")
RATINGS_CSV_LOCAL = os.path.join(DATA_DIR, "ratings.csv")

ML_LATEST_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ML_LATEST_SMALL_DIR = os.path.join(DATA_DIR, "ml-latest-small")

MOVIELENS_MOVIE_COLS = frozenset({"movieId", "title", "genres"})
MOVIELENS_RATING_COLS = frozenset({"userId", "movieId", "rating"})

USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_]{3,32}$")

TMDB_API_KEY = os.environ.get("TMDB_API_KEY")

MOVIE_SYNOPSES = {
    "Toy Story (1995)": "A cowboy doll feels threatened when a new spaceman toy becomes Andy's favorite.",
    "Jumanji (1995)": "Two children discover a magical board game that brings jungle dangers into the real world.",
    "Heat (1995)": "A master thief and a detective become obsessed with outsmarting each other.",
    "The Matrix (1999)": "A hacker discovers reality is a simulated world controlled by machines.",
    "Forrest Gump (1994)": "The extraordinary life journey of a kind-hearted man unfolds across decades.",
    "default": "A compelling film praised for its storytelling and performances."
}

MOOD_GENRE_MAP: Dict[str, List[str]] = {
    "Happy 😊": ["Comedy", "Animation", "Family"],
    "Tense 🎬": ["Thriller", "Crime", "Mystery"],
    "Thoughtful 🧠": ["Drama", "Documentary", "History"],
    "Adventurous 🚀": ["Action", "Adventure", "Sci-Fi"],
    "Romantic 💕": ["Romance"],
}

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color:#071021; color:#e6eef8; }
.block-container { padding-top: 1.2rem; max-width: 1200px; }
.header-title { font-size: 2.4rem; font-weight:800; text-align:center; margin-bottom:0.1rem; }
.header-sub { text-align:center; color:#94a3b8; margin-bottom:1rem; }
.movie-card { background:#0f172a; border:1px solid #1e293b; border-radius:12px; padding:14px; margin-bottom:12px; }
.movie-title { font-size:1.05rem; font-weight:700; color:#fff; margin-bottom:6px; }
.movie-genre { color:#60a5fa; font-size:0.9rem; margin-bottom:8px; }
.movie-desc { color:#cbd5e1; font-size:0.9rem; min-height:64px; }
.stButton > button { border-radius:8px; height:40px; }
.login-box { max-width:520px; margin:auto; padding:22px; border-radius:14px; background:rgba(10,15,25,0.9); border:1px solid rgba(255,255,255,0.04); }
.footer { text-align:center; color:#64748b; margin-top:28px; padding:18px; font-size:13px; }
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ---------------- Layer: Database ----------------
def _db_connect() -> sqlite3.Connection:
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
                favorite_genres TEXT,
                favorite_movie_hint TEXT,
                onboarding_mood TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS watch_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                movie_title TEXT NOT NULL,
                watched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_watch_unique
            ON watch_history(username, movie_title)
        """)

# ---------------- Layer: Validation / Auth ----------------
def _validate_password_strength(password: str) -> Optional[str]:
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if not any(c.isupper() for c in password):
        return "Password must contain at least one uppercase letter."
    if not any(c.islower() for c in password):
        return "Password must contain at least one lowercase letter."
    if not any(c.isdigit() for c in password):
        return "Password must contain at least one number."
    if not any(c in "!@#$%^&*()-_=+[]{};:,.<>/?\\|`~" for c in password):
        return "Password must contain at least one special character."
    return None

def _validate_username_str(username: str) -> Optional[str]:
    u = (username or "").strip()
    if len(u) < 3:
        return "Username must be at least 3 characters."
    if not USERNAME_PATTERN.match(u):
        return "Use only letters, digits, or underscore."
    return None

def _signup(
    username: str,
    password: str,
    confirm: str,
    favorite_genres: Optional[List[str]] = None,
    favorite_movie_hint: str = "",
    onboarding_mood: str = "",
) -> Tuple[bool, str]:
    err = _validate_username_str(username)
    if err:
        return False, err
    if password != confirm:
        return False, "Passwords do not match."
    if "\x00" in password:
        return False, "Password contains invalid characters."
    err = _validate_password_strength(password)
    if err:
        return False, err

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    fg = "|".join(favorite_genres) if favorite_genres else None
    fm = (favorite_movie_hint or "").strip()[:180] or None
    om = (onboarding_mood or "").strip() or None
    if om == "—":
        om = None

    try:
        with _db_connect() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, favorite_genres, favorite_movie_hint, onboarding_mood) VALUES (?, ?, ?, ?, ?)",
                (username.strip().lower(), hashed, fg, fm, om),
            )
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "Username already taken — please choose another."
    except sqlite3.Error:
        return False, "Database error while creating account."

def _login(username: str, password: str) -> Tuple[bool, str]:
    err = _validate_username_str(username)
    if err:
        return False, err
    try:
        with _db_connect() as conn:
            row = conn.execute("SELECT password_hash FROM users WHERE username = ?", (username.strip().lower(),)).fetchone()
    except sqlite3.Error:
        return False, "Could not read user database."
    if not row:
        return False, "Username not found."
    try:
        ok = bcrypt.checkpw(password.encode(), row[0].encode())
    except ValueError:
        return False, "Account data is invalid — please sign up again."
    if ok:
        return True, "Login successful."
    return False, "Incorrect password."

# ---------------- Layer: Watch History ----------------
def watch_history_push(username: str, movie_title: str) -> None:
    if username == "guest" or not movie_title:
        return
    try:
        with _db_connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO watch_history (username, movie_title) VALUES (?, ?)",
                (username.strip().lower(), movie_title),
            )
    except sqlite3.Error:
        pass

def watch_history_list(username: str, limit: int = 10) -> List[str]:
    if username == "guest":
        return []
    try:
        with _db_connect() as conn:
            rows = conn.execute(
                "SELECT movie_title FROM watch_history WHERE username = ? ORDER BY watched_at DESC LIMIT ?",
                (username.strip().lower(), limit),
            ).fetchall()
    except sqlite3.Error:
        return []
    seen: Set[str] = set()
    out: List[str] = []
    for (t,) in rows:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out

# ---------------- Layer: Data Handling ----------------
def _ensure_data_downloaded() -> Tuple[str, str]:
    if os.path.exists(MOVIES_CSV_LOCAL) and os.path.exists(RATINGS_CSV_LOCAL):
        return MOVIES_CSV_LOCAL, RATINGS_CSV_LOCAL
    os.makedirs(DATA_DIR, exist_ok=True)
    em = os.path.join(ML_LATEST_SMALL_DIR, "movies.csv")
    er = os.path.join(ML_LATEST_SMALL_DIR, "ratings.csv")
    if os.path.exists(em) and os.path.exists(er):
        return em, er
    zp = os.path.join(DATA_DIR, "ml-latest-small.zip")
    try:
        with st.spinner("Downloading MovieLens dataset..."):
            urllib.request.urlretrieve(ML_LATEST_SMALL_URL, zp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"MovieLens download failed (HTTP {e.code}).") from e
    except urllib.error.URLError:
        raise RuntimeError("MovieLens download failed (no network). Place CSVs under Data/MovieLens/.")
    except OSError as e:
        raise RuntimeError(f"MovieLens download failed (disk error): {e}") from e
    try:
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(DATA_DIR)
    except zipfile.BadZipFile:
        raise RuntimeError("Downloaded archive is corrupted. Delete ml-latest-small.zip and retry.")
    if not (os.path.exists(em) and os.path.exists(er)):
        raise FileNotFoundError("Download finished but CSVs were not found in the archive.")
    return em, er

def _read_csv_checked(path: str, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (ParserError, EmptyDataError) as e:
        raise ValueError(f"{label}: CSV is empty or could not be parsed ({path}): {e}") from e
    except (OSError, UnicodeDecodeError) as e:
        raise ValueError(f"{label}: could not read file ({path}): {e}") from e

@st.cache_data(show_spinner=False)
def load_and_prepare_data(top_n_movies: int = 500, top_n_users: int = 500):
    mp, rp = _ensure_data_downloaded()
    movies = _read_csv_checked(mp, "movies.csv")
    ratings = _read_csv_checked(rp, "ratings.csv")

    m_miss = MOVIELENS_MOVIE_COLS - set(movies.columns)
    r_miss = MOVIELENS_RATING_COLS - set(ratings.columns)
    if m_miss:
        raise ValueError(f"movies.csv missing columns {sorted(m_miss)}.")
    if r_miss:
        raise ValueError(f"ratings.csv missing columns {sorted(r_miss)}.")

    try:
        df = pd.merge(ratings, movies, on="movieId", how="inner")
    except KeyError as e:
        raise ValueError(f"Could not merge on movieId: {e}") from e

    df.dropna(inplace=True)
    if df.empty:
        raise ValueError("Merged data is empty after dropping nulls.")

    try:
        df = df[df["title"].isin(df["title"].value_counts().head(top_n_movies).index)]
        df = df[df["userId"].isin(df["userId"].value_counts().head(top_n_users).index)]
    except (KeyError, TypeError) as e:
        raise ValueError(f"Could not filter by popularity: {e}") from e

    if df.empty:
        raise ValueError("No ratings left after filtering. Lower the sliders and rebuild.")

    try:
        gmap = movies[["title", "genres"]].drop_duplicates("title").set_index("title")["genres"].to_dict()
        umm = df.pivot_table(index="userId", columns="title", values="rating", aggfunc="mean").fillna(0)
    except (KeyError, ValueError) as e:
        raise ValueError(f"Could not build matrix: {e}") from e

    return df, movies, umm, gmap

# ---------------- Layer: TMDB (optional poster fetch) ----------------
def fetch_movie_poster(title: str) -> Optional[str]:
    if not TMDB_API_KEY:
        return None
    try:
        q = urllib.parse.quote(title)
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={q}"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results")
        if results:
            poster_path = results[0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        pass
    return None

# ---------------- Layer: Recommendation ----------------
def recommend_for_user(username: str, df: pd.DataFrame, movies_df: pd.DataFrame, umm: pd.DataFrame, gmap: Dict[str, str], top_k: int = 6) -> List[str]:
    if username == "guest":
        return df.groupby("title")["rating"].mean().sort_values(ascending=False).head(top_k).index.tolist()

    history = watch_history_list(username)
    if not history:
        return df["title"].value_counts().head(top_k).index.tolist()

    seen = set(history)
    liked_genres: Set[str] = set()
    for movie in history:
        genres = gmap.get(movie)
        if genres:
            liked_genres.update(genres.split("|"))

    preferred_mood = None
    try:
        with _db_connect() as conn:
            row = conn.execute("SELECT onboarding_mood FROM users WHERE username = ?", (username.strip().lower(),)).fetchone()
            if row:
                preferred_mood = row[0]
    except sqlite3.Error:
        preferred_mood = None

    recommendation_scores: List[Tuple[str, float]] = []
    popularity_map = df.groupby("title")["rating"].mean().to_dict()

    for title, genres in gmap.items():
        if title in seen:
            continue

        genre_score = sum(1 for g in liked_genres if g in genres)
        popularity = float(popularity_map.get(title, 0.0))
        mood_bonus = 0.0
        if preferred_mood and preferred_mood in MOOD_GENRE_MAP:
            for mg in MOOD_GENRE_MAP[preferred_mood]:
                if mg in genres:
                    mood_bonus += 1.5

        total_score = genre_score + popularity + mood_bonus
        recommendation_scores.append((title, total_score))

    recommendation_scores.sort(key=lambda x: x[1], reverse=True)
    return [title for title, _ in recommendation_scores[:top_k]]

# ---------------- Layer: UI / Auth Screen ----------------
def render_auth_screen() -> bool:
    if st.session_state.get("logged_in"):
        return True

    _init_auth_db()

    st.markdown("<div class='login-box'>", unsafe_allow_html=True)
    st.markdown("<div class='header-title'>🎬 RecoMind</div>", unsafe_allow_html=True)
    st.markdown("<div class='header-sub'>AI-powered emotional movie recommendation system</div>", unsafe_allow_html=True)

    auth_mode = st.radio("Account", ["Login", "Sign Up"], horizontal=True, label_visibility="collapsed")

    if auth_mode == "Login":
        uname = st.text_input("Username", key="login_user")
        pwd = st.text_input("Password", type="password", key="login_pwd")
        col1, col2 = st.columns([3, 2])
        with col1:
            if st.button("Login", use_container_width=True):
                ok, msg = _login(uname, pwd)
                if ok:
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = uname.strip().lower()
                    st.rerun()
                else:
                    st.error(msg)
        with col2:
            if st.button("👤 Guest Demo", use_container_width=True):
                st.session_state["logged_in"] = True
                st.session_state["username"] = "guest"
                st.rerun()

    else:
        new_user = st.text_input("Choose Username", key="signup_user", placeholder="e.g. smit_patel")
        st.caption("3–32 characters: letters, digits, or underscore only.")
        new_pwd = st.text_input("Choose Password (min 8 chars, 1 upper, 1 lower, 1 number, 1 special)", type="password", key="signup_pwd")
        confirm_pwd = st.text_input("Confirm Password", type="password", key="signup_confirm")

        st.markdown("**Cold start profile (optional)**")
        fav_gen = st.multiselect("Favourite genres", sorted({g for gl in MOOD_GENRE_MAP.values() for g in gl}), key="signup_fav_gen")
        fav_movie = st.text_input("A favourite movie (partial title is OK)", key="signup_fav_movie", max_chars=180)
        mood_pre = st.selectbox("Preferred mood vibe", ["—"] + list(MOOD_GENRE_MAP.keys()), key="signup_mood")

        if st.button("Create Account", use_container_width=True):
            ok, msg = _signup(new_user, new_pwd, confirm_pwd, fav_gen or None, fav_movie, mood_pre)
            if ok:
                st.success(msg + " Please log in.")
            else:
                st.error(msg)

    st.markdown("</div>", unsafe_allow_html=True)
    return False

# ---------------- Layer: Main App ----------------
def main():
    logged = render_auth_screen()
    if not logged:
        return

    username = st.session_state.get("username", "guest")
    if username == "guest":
        st.markdown("<div class='header-title'>🎬 Welcome to RecoMind</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='header-title'>🎬 Welcome, {username}</div>", unsafe_allow_html=True)

    st.caption("Collaborative filtering · Mood-aware · Wellbeing filter")

    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()

    with st.sidebar:
        st.header("Filters")
        top_n_movies = st.slider("Top N movies", 100, 2000, 500, step=100)
        top_n_users = st.slider("Top N users", 100, 2000, 500, step=100)

    try:
        df, movies_df, umm, gmap = load_and_prepare_data(top_n_movies, top_n_users)
    except Exception as e:
        st.error(f"Dataset loading failed: {e}")
        return

    recs = recommend_for_user(username, df, movies_df, umm, gmap, top_k=6)

    st.markdown("---")
    st.subheader("Recommended for you")

    cols = st.columns(3)
    for i, title in enumerate(recs):
        with cols[i % 3]:
            poster = fetch_movie_poster(title)
            if poster:
                st.image(poster, use_column_width=True)
            st.markdown(
                f"<div class='movie-card'><div class='movie-title'>{title}</div>"
                f"<div class='movie-genre'>🎭 {gmap.get(title, 'Unknown')}</div>"
                f"<div class='movie-desc'>{MOVIE_SYNOPSES.get(title, MOVIE_SYNOPSES['default'])}</div></div>",
                unsafe_allow_html=True,
            )
            if st.button("Mark Watched", key=f"watch_{i}", use_container_width=True):
                watch_history_push(username, title)
                st.success(f"{title} added to history.")
                st.rerun()

    st.markdown("---")
    st.subheader("Your Recent Watch History")
    hist = watch_history_list(username)
    if hist:
        for item in hist:
            st.write(f"• {item}")
    else:
        st.info("No watch history available.")

    st.markdown("---")
    st.markdown("<div class='footer'>RecoMind · Built with Streamlit · MovieLens</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
