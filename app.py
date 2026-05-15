import os
import re
import sqlite3
import zipfile
import urllib.error
import urllib.request
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import bcrypt
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
import streamlit as st

st.set_page_config(
    page_title="RecoMind",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "Data", "MovieLens")

MOVIES_CSV_LOCAL = os.path.join(DATA_DIR, "movies.csv")
RATINGS_CSV_LOCAL = os.path.join(DATA_DIR, "ratings.csv")

FILTERED_DATA_CSV_LOCAL = os.path.join(DATA_DIR, "filtered_data.csv")
FILTERED_MOVIES_DATA_CSV_LOCAL = os.path.join(DATA_DIR, "filtered_movies_data.csv")

ML_LATEST_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ML_LATEST_SMALL_DIR = os.path.join(DATA_DIR, "ml-latest-small")

MOVIELENS_MOVIE_COLS = frozenset({"movieId", "title", "genres"})
MOVIELENS_RATING_COLS = frozenset({"userId", "movieId", "rating"})

USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_]{3,32}$")

COLD_START_GENRE_OPTIONS = sorted({
    "Animation", "Comedy", "Family", "Fantasy", "Adventure", "Musical",
    "Romance", "Horror", "Thriller", "Crime", "War", "Film-Noir",
    "Action", "Drama", "Sci-Fi", "Mystery", "Documentary", "History",
    "Western", "IMAX"
})

MOOD_GENRE_MAP = {
    "Happy 😊": {"Comedy", "Family", "Animation"},
    "Tense 🎬": {"Thriller", "Crime", "Mystery"},
    "Thoughtful 🧠": {"Drama", "Documentary", "History"},
    "Adventurous 🚀": {"Action", "Adventure", "Sci-Fi"},
    "Romantic 💕": {"Romance"},
}

MOVIE_SYNOPSES = {
    "Toy Story (1995)": "A cowboy doll is profoundly threatened and jealous when a new spaceman action figure supplants him as top toy in a boy's bedroom.",
    "Pulp Fiction (1994)": "The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption.",
    "The Shawshank Redemption (1994)": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
    "Forrest Gump (1994)": "The history of the United States from the 1950s to the 1970s unfolds from the perspective of an Alabama man with an IQ of 75.",
    "The Matrix (1999)": "When a beautiful stranger leads computer hacker Neo to a forbidding underworld, he discovers the shocking truth about the life he knows.",
    "default": "A compelling film that takes audiences on an unforgettable journey. Widely praised for its storytelling, performances, and direction.",
}

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #071021;
    color: #e6eef8;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

.login-box {
    width: 100%;
    max-width: 520px;
    margin: auto;
    padding: 30px;
    border-radius: 20px;
    background: rgba(10,15,25,0.92);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 0 30px rgba(0,0,0,0.35);
}

.login-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    color: white;
}

.login-subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 20px;
}

.movie-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 20px;
    height: 100%;
}

.movie-title {
    font-size: 20px;
    font-weight: 700;
    color: white;
    margin-bottom: 10px;
}

.movie-genre {
    color: #38bdf8;
    font-size: 14px;
    margin-bottom: 12px;
}

.movie-description {
    color: #cbd5e1;
    font-size: 14px;
    line-height: 1.6;
    min-height: 120px;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
    font-size: 15px;
}

hr {
    border-color: #1e293b;
}
</style>
"""

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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        try:
            conn.execute("ALTER TABLE users ADD COLUMN favorite_genres TEXT")
        except sqlite3.OperationalError:
            pass

        try:
            conn.execute("ALTER TABLE users ADD COLUMN favorite_movie_hint TEXT")
        except sqlite3.OperationalError:
            pass

        try:
            conn.execute("ALTER TABLE users ADD COLUMN onboarding_mood TEXT")
        except sqlite3.OperationalError:
            pass

        conn.execute("""
            CREATE TABLE IF NOT EXISTS watch_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                movie_title TEXT NOT NULL,
                watched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

def _validate_password_strength(password: str) -> Optional[str]:
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if not any(c.isupper() for c in password):
        return "Password must contain at least one uppercase letter."
    if not any(c.isdigit() for c in password):
        return "Password must contain at least one number."
    return None

def _validate_username_str(username: str) -> Optional[str]:
    u = username.strip()

    if len(u) < 3:
        return "Username must be at least 3 characters."

    if not USERNAME_PATTERN.match(u):
        return "Use only letters, digits, or underscore."

    return None

def _signup(username, password, confirm, favorite_genres=None, favorite_movie_hint="", onboarding_mood=""):
    err = _validate_username_str(username)

    if err:
        return False, err

    if password != confirm:
        return False, "Passwords do not match."

    err = _validate_password_strength(password)

    if err:
        return False, err

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    fg = "|".join(favorite_genres) if favorite_genres else None

    try:
        with _db_connect() as conn:
            conn.execute("""
                INSERT INTO users
                (username, password_hash, favorite_genres, favorite_movie_hint, onboarding_mood)
                VALUES (?, ?, ?, ?, ?)
            """, (
                username.strip().lower(),
                hashed,
                fg,
                favorite_movie_hint,
                onboarding_mood
            ))

        return True, "Account created successfully."

    except sqlite3.IntegrityError:
        return False, "Username already exists."

def _login(username: str, password: str):
    try:
        with _db_connect() as conn:
            row = conn.execute(
                "SELECT password_hash FROM users WHERE username=?",
                (username.strip().lower(),)
            ).fetchone()

        if not row:
            return False, "Username not found."

        ok = bcrypt.checkpw(password.encode(), row[0].encode())

        if ok:
            return True, "Login successful."

        return False, "Incorrect password."

    except sqlite3.Error as e:
        return False, str(e)

def watch_history_list(username: str, limit: int = 10):
    if username == "guest":
        return []

    try:
        with _db_connect() as conn:
            rows = conn.execute("""
                SELECT movie_title
                FROM watch_history
                WHERE username=?
                ORDER BY watched_at DESC
                LIMIT ?
            """, (username, limit)).fetchall()

        return [r[0] for r in rows]

    except sqlite3.Error:
        return []

def watch_history_push(username: str, movie_title: str):
    if username == "guest":
        return

    try:
        with _db_connect() as conn:
            conn.execute("""
                INSERT INTO watch_history (username, movie_title)
                VALUES (?, ?)
            """, (username, movie_title))
    except sqlite3.Error:
        pass

def render_auth_screen():
    if st.session_state.get("logged_in"):
        return True

    _init_auth_db()

    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='login-box'>", unsafe_allow_html=True)

        st.markdown("<div class='login-title'>🎬 RecoMind</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='login-subtitle'>AI-powered emotional movie recommendation system</div>",
            unsafe_allow_html=True
        )

        auth_mode = st.radio(
            "Account",
            ["Login", "Sign Up"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if auth_mode == "Login":

            uname = st.text_input("Username")
            pwd = st.text_input("Password", type="password")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Login"):
                    ok, msg = _login(uname, pwd)

                    if ok:
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = uname
                        st.rerun()
                    else:
                        st.error(msg)

            with col2:
                if st.button("Guest Mode"):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = "guest"
                    st.rerun()

        else:
            new_user = st.text_input("Choose Username")
            new_pwd = st.text_input("Choose Password", type="password")
            confirm_pwd = st.text_input("Confirm Password", type="password")

            fav_gen = st.multiselect(
                "Favourite Genres",
                COLD_START_GENRE_OPTIONS
            )

            fav_movie = st.text_input("Favourite Movie")

            mood_pre = st.selectbox(
                "Preferred Mood",
                [""] + list(MOOD_GENRE_MAP.keys())
            )

            if st.button("Create Account"):
                ok, msg = _signup(
                    new_user,
                    new_pwd,
                    confirm_pwd,
                    fav_gen,
                    fav_movie,
                    mood_pre
                )

                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

        st.markdown("</div>", unsafe_allow_html=True)

    return False

def _ensure_data_downloaded():
    if os.path.exists(MOVIES_CSV_LOCAL) and os.path.exists(RATINGS_CSV_LOCAL):
        return MOVIES_CSV_LOCAL, RATINGS_CSV_LOCAL

    os.makedirs(DATA_DIR, exist_ok=True)

    em = os.path.join(ML_LATEST_SMALL_DIR, "movies.csv")
    er = os.path.join(ML_LATEST_SMALL_DIR, "ratings.csv")

    if os.path.exists(em) and os.path.exists(er):
        return em, er

    zp = os.path.join(DATA_DIR, "ml-latest-small.zip")

    with st.spinner("Downloading MovieLens dataset..."):
        urllib.request.urlretrieve(ML_LATEST_SMALL_URL, zp)

    with zipfile.ZipFile(zp, "r") as zf:
        zf.extractall(DATA_DIR)

    return em, er

def _read_csv_checked(path: str, label: str):
    try:
        return pd.read_csv(path)

    except (ParserError, EmptyDataError) as e:
        raise ValueError(f"{label} parsing error: {e}")

@st.cache_data(show_spinner=False)
def load_and_prepare_data(top_n_movies=500, top_n_users=500):

    mp, rp = _ensure_data_downloaded()

    movies = _read_csv_checked(mp, "movies.csv")
    ratings = _read_csv_checked(rp, "ratings.csv")

    df = pd.merge(ratings, movies, on="movieId", how="inner")

    df = df[df["title"].isin(
        df["title"].value_counts().head(top_n_movies).index
    )]

    df = df[df["userId"].isin(
        df["userId"].value_counts().head(top_n_users).index
    )]

    gmap = movies[["title", "genres"]].drop_duplicates(
        "title"
    ).set_index("title")["genres"].to_dict()

    umm = df.pivot_table(
        index="userId",
        columns="title",
        values="rating",
        aggfunc="mean"
    ).fillna(0)

    return df, movies, umm, gmap

def recommend_for_user(username, df, movies_df, umm, gmap, top_k=6):

    if username == "guest":
        return df.groupby("title")["rating"].mean().sort_values(
            ascending=False
        ).head(top_k).index.tolist()

    history = watch_history_list(username)

    if history:
        seen = set(history)
        genres = []

        for t in history:
            g = gmap.get(t)

            if g:
                genres.extend(g.split("|"))

        genres = set(genres)

        candidates = []

        for title, g in gmap.items():

            if title in seen:
                continue

            score = sum(1 for gg in genres if gg in g)

            if score > 0:
                candidates.append((title, score))

        if candidates:
            candidates = sorted(
                candidates,
                key=lambda x: x[1],
                reverse=True
            )

            return [title for title, _ in candidates[:top_k]]

    return df["title"].value_counts().head(top_k).index.tolist()

def main():

    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    logged = render_auth_screen()

    if not logged:
        return

    username = st.session_state.get("username", "guest")

    if username == "guest":
        st.title("🎬 Welcome to RecoMind")
    else:
        st.title(f"🎬 Welcome, {username}")

    st.caption(
        "Collaborative filtering · Mood-aware · Wellbeing filter · TextBlob NLP · TMDB"
    )

    col1, col2 = st.columns([8, 1])

    with col2:
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()

    with st.sidebar:
        st.header("Filters")

        top_n_movies = st.slider(
            "Top N movies",
            100,
            2000,
            500,
            step=100
        )

        top_n_users = st.slider(
            "Top N users",
            100,
            2000,
            500,
            step=100
        )

    try:
        df, movies_df, umm, gmap = load_and_prepare_data(
            top_n_movies,
            top_n_users
        )

    except Exception as e:
        st.error(f"Dataset loading failed: {e}")
        return

    recs = recommend_for_user(
        username,
        df,
        movies_df,
        umm,
        gmap,
        top_k=6
    )

    st.markdown("---")
    st.subheader("Recommended for you")

    cols = st.columns(3)

    for i, title in enumerate(recs):

        with cols[i % 3]:

            st.markdown("<div class='movie-card'>", unsafe_allow_html=True)

            st.markdown(
                f"<div class='movie-title'>{title}</div>",
                unsafe_allow_html=True
            )

            genres = gmap.get(title, "Unknown Genre")

            st.markdown(
                f"<div class='movie-genre'>🎭 {genres}</div>",
                unsafe_allow_html=True
            )

            synopsis = MOVIE_SYNOPSES.get(
                title,
                MOVIE_SYNOPSES["default"]
            )

            st.markdown(
                f"<div class='movie-description'>{synopsis}</div>",
                unsafe_allow_html=True
            )

            if st.button(
                f"Mark Watched",
                key=f"watch_{i}"
            ):
                watch_history_push(username, title)
                st.success(f"{title} added to history.")
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Your Recent Watch History")

    hist = watch_history_list(username)

    if hist:
        for item in hist:
            st.write(f"• {item}")
    else:
        st.info("No watch history available.")

    st.markdown("---")

    st.markdown(
        """
        <div style='text-align:center;color:#64748b;padding:20px;font-size:14px;'>
        RecoMind · Built by Smit Patel 🚀 · Streamlit · MovieLens
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
