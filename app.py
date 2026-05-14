# app.py
import html
import os
import re
import sqlite3
import zipfile
import urllib.error
import urllib.request
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import bcrypt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────
#  Constants / Data paths / Small fixtures
# ─────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "Data", "MovieLens")

MOVIES_CSV_LOCAL = os.path.join(DATA_DIR, "movies.csv")
RATINGS_CSV_LOCAL = os.path.join(DATA_DIR, "ratings.csv")

FILTERED_DATA_CSV_LOCAL = os.path.join(DATA_DIR, "filtered_data.csv")
FILTERED_MOVIES_DATA_CSV_LOCAL = os.path.join(DATA_DIR, "filtered_movies_data.csv")
ML_LATEST_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ML_LATEST_SMALL_DIR = os.path.join(DATA_DIR, "ml-latest-small")

_POSTER_FALLBACK_HTML = (
    "<div style='background:#0d1117;border:1px solid #1f2937;border-radius:8px;"
    "height:200px;display:flex;align-items:center;justify-content:center;"
    "color:#374151;font-size:2.5rem;'>🎬</div>"
)

MOVIELENS_MOVIE_COLS = frozenset({"movieId", "title", "genres"})
MOVIELENS_RATING_COLS = frozenset({"userId", "movieId", "rating"})

USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_]{3,32}$")
COLD_START_GENRE_OPTIONS = sorted({
    "Animation", "Comedy", "Family", "Fantasy", "Adventure", "Musical",
    "Romance", "Horror", "Thriller", "Crime", "War", "Film-Noir",
    "Action", "Drama", "Sci-Fi", "Mystery", "Documentary", "History",
    "Western", "IMAX"
})

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
    "default": "A compelling film that takes audiences on an unforgettable journey. Widely praised for its storytelling, performances, and direction.",
}

# ─────────────────────────────────────────────────────────────────
#  CSS and small HTML snippets
# ─────────────────────────────────────────────────────────────────
DARK_CARD_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Space+Mono:wght@400;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.auth-brand-title { text-align:center; font-size:3rem; font-weight:800; color:#f8fafc; line-height:1.15; margin:0 0 .35rem 0; white-space:nowrap; }
</style>"""

SKELETON_HTML = """
<div style="background:#0d1117;border:1px solid #1f2937;border-radius:10px;padding:.95rem 1.15rem;margin-bottom:.55rem;">
  <div style="background:linear-gradient(90deg,#111827 25%,#1f2937 50%,#111827 75%);height:17px;border-radius:6px;margin-bottom:.6rem;"></div>
  <div style="background:linear-gradient(90deg,#111827 25%,#1f2937 50%,#111827 75%);height:13px;border-radius:6px;margin-bottom:.45rem;width:55%;"></div>
  <div style="background:linear-gradient(90deg,#111827 25%,#1f2937 50%,#111827 75%);height:12px;border-radius:6px;margin-bottom:.35rem;"></div>
  <div style="background:linear-gradient(90deg,#111827 25%,#1f2937 50%,#111827 75%);height:12px;border-radius:6px;width:72%;"></div>
</div>
"""

# ─────────────────────────────────────────────────────────────────
#  Auth / DB helpers
# ─────────────────────────────────────────────────────────────────
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
        # add optional columns if missing
        for col_sql in (
            "ALTER TABLE users ADD COLUMN favorite_genres TEXT",
            "ALTER TABLE users ADD COLUMN favorite_movie_hint TEXT",
            "ALTER TABLE users ADD COLUMN onboarding_mood TEXT",
        ):
            try:
                conn.execute(col_sql)
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
        conn.execute("CREATE INDEX IF NOT EXISTS idx_watch_user_time ON watch_history(username, watched_at)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rec_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                movie_title TEXT NOT NULL,
                anchor_title TEXT,
                sentiment TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
        return "Use 3–32 characters: letters, digits, or underscore only."
    return None


def _signup(
    username: str,
    password: str,
    confirm: str,
    favorite_genres: Optional[List[str]] = None,
    favorite_movie_hint: str = "",
    onboarding_mood: str = "",
) -> tuple:
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
    # bcrypt returns bytes; store as utf-8 string
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode("utf-8", errors="ignore")
    fg = "|".join(favorite_genres) if favorite_genres else None
    fm = favorite_movie_hint.strip()[:180] or None
    om = (onboarding_mood or "").strip() or None
    if om == "—":
        om = None
    try:
        with _db_connect() as conn:
            conn.execute(
                """
                INSERT INTO users (username, password_hash, favorite_genres, favorite_movie_hint, onboarding_mood)
                VALUES (?, ?, ?, ?, ?)
                """,
                (username.strip().lower(), hashed, fg, fm, om),
            )
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "Username already taken — please choose another."
    except sqlite3.Error as e:
        return False, f"Database error while creating account: {e}"


def _login(username: str, password: str) -> tuple:
    err = _validate_username_str(username)
    if err:
        return False, err
    try:
        with _db_connect() as conn:
            row = conn.execute(
                "SELECT password_hash FROM users WHERE username = ?",
                (username.strip().lower(),),
            ).fetchone()
    except sqlite3.Error as e:
        return False, f"Could not read user database: {e}"
    if not row:
        return False, "Username not found."
    try:
        ok = bcrypt.checkpw(password.encode(), row[0].encode())
    except ValueError:
        return False, "Account data is invalid — please sign up again."
    if ok:
        return True, "Login successful."
    return False, "Incorrect password."


def watch_history_list(username: str, limit: int = 40) -> List[str]:
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
    return out[:5]


def watch_history_push(username: str, movie_title: str) -> None:
    if username == "guest" or not movie_title:
        return
    try:
        with _db_connect() as conn:
            conn.execute(
                "INSERT INTO watch_history (username, movie_title) VALUES (?, ?)",
                (username.strip().lower(), movie_title),
            )
    except sqlite3.Error:
        # intentionally silent in UI; consider logging in production
        pass


def rec_feedback_push(username: str, movie_title: str, anchor_title: str, sentiment: str) -> None:
    if username == "guest" or not movie_title or not sentiment:
        return
    try:
        with _db_connect() as conn:
            conn.execute(
                "INSERT INTO rec_feedback (username, movie_title, anchor_title, sentiment) VALUES (?, ?, ?, ?)",
                (username.strip().lower(), movie_title, anchor_title or "", sentiment),
            )
    except sqlite3.Error:
        pass


# ─────────────────────────────────────────────────────────────────
#  UI: Authentication screen
# ─────────────────────────────────────────────────────────────────
def render_auth_screen() -> bool:
    """Returns True if user is logged in."""
    if st.session_state.get("logged_in"):
        return True

    _init_auth_db()

    # Inject CSS and small layout fixes
    st.markdown(DARK_CARD_CSS, unsafe_allow_html=True)
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #0d1117; }
    [data-testid="stHeader"] { background-color: transparent; }
    section.main > div { min-height: calc(100vh - 5rem); display:flex; flex-direction:column; justify-content:center; align-items:stretch; padding-top:0 !important; }
    .block-container { padding-top:0.5rem !important; padding-bottom:2rem !important; max-width:680px !important; min-width:420px !important; margin-left:auto !important; margin-right:auto !important; }
    @media (max-width:768px) { .block-container { min-width:unset !important; width:95% !important; padding-left:1rem !important; padding-right:1rem !important; } .auth-brand-title { font-size:2.2rem !important; } }
    </style>
    """, unsafe_allow_html=True)

    _, center_col, _ = st.columns([1, 4, 1])
    with center_col:
        try:
            ctx = st.container(border=True)
        except TypeError:
            ctx = st.container()
        with ctx:
            st.markdown("<div class='auth-brand-title'>🎬 RecoMind</div>", unsafe_allow_html=True)
            st.markdown(
                "<div style='text-align:center;font-size:0.88rem;color:#64748b;margin:0 0 0.85rem 0;line-height:1.45;'>"
                "AI-powered emotional movie recommendation system</div>",
                unsafe_allow_html=True,
            )

            auth_mode = st.radio(
                "Account",
                ["Login", "Sign Up"],
                horizontal=True,
                key="auth_mode_switch",
                label_visibility="collapsed",
            )

            if auth_mode == "Login":
                uname = st.text_input("Username", key="login_user", placeholder="your_username")
                pwd   = st.text_input("Password", type="password", key="login_pwd")
                st.caption("Letters, digits, underscore · 3–32 characters.")

                col_login, col_guest = st.columns([3, 2])

                with col_login:
                    if st.button("Login", use_container_width=True):
                        ok, msg = _login(uname, pwd)
                        if ok:
                            st.session_state["logged_in"] = True
                            st.session_state["username"]  = uname.strip().lower()
                            st.rerun()
                        else:
                            st.error(msg)

                with col_guest:
                    if st.button("👤 Guest Demo", use_container_width=True):
                        st.session_state["logged_in"] = True
                        st.session_state["username"]  = "guest"
                        st.rerun()

            else:
                new_user    = st.text_input("Choose Username", key="signup_user", placeholder="e.g. smit_patel")
                st.caption("3–32 characters: letters, digits, or underscore only.")
                new_pwd     = st.text_input(
                    "Choose Password (min 8 chars, 1 uppercase, 1 number)",
                    type="password", key="signup_pwd",
                )
                confirm_pwd = st.text_input("Confirm Password", type="password", key="signup_confirm")

                st.markdown("**Cold start profile (optional)**")
                st.caption("Helps mitigate the new-user cold-start problem before you have watch history.")
                fav_gen   = st.multiselect("Favourite genres", COLD_START_GENRE_OPTIONS, key="signup_fav_gen")
                fav_movie = st.text_input("A favourite movie (partial title is OK)", key="signup_fav_movie", max_chars=180)
                mood_pre  = st.selectbox("Preferred mood vibe", ["—"] + list(MOOD_GENRE_MAP.keys()), key="signup_mood")

                if st.button("Create Account", use_container_width=True):
                    ok, msg = _signup(new_user, new_pwd, confirm_pwd, fav_gen or None, fav_movie, mood_pre)
                    if ok:
                        st.success(msg + " Please log in.")
                    else:
                        st.error(msg)

    return False


# ─────────────────────────────────────────────────────────────────
#  Data layer: download, read, prepare
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
    try:
        with st.spinner("Downloading MovieLens dataset…"):
            urllib.request.urlretrieve(ML_LATEST_SMALL_URL, zp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"MovieLens download failed (HTTP {e.code}).") from e
    except urllib.error.URLError as e:
        raise RuntimeError("MovieLens download failed (no network). Place CSVs under Data/MovieLens/.") from e
    except OSError as e:
        raise RuntimeError(f"MovieLens download failed (disk error): {e}") from e
    try:
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(DATA_DIR)
    except zipfile.BadZipFile as e:
        raise RuntimeError("Downloaded archive is corrupted. Delete ml-latest-small.zip and retry.") from e
    except OSError as e:
        raise RuntimeError(f"Could not extract archive: {e}") from e
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
    """
    Returns: df (merged ratings+movies), movies_df (unique movies), user_movie_matrix (pivot), genre_map (title->genres)
    """
    fs = None
    if os.path.exists(FILTERED_MOVIES_DATA_CSV_LOCAL):
        fs = FILTERED_MOVIES_DATA_CSV_LOCAL
    elif os.path.exists(FILTERED_DATA_CSV_LOCAL):
        fs = FILTERED_DATA_CSV_LOCAL
    if fs is not None:
        df = _read_csv_checked(fs, "Filtered dataset")
        need = {"userId", "title", "rating"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"Filtered CSV missing columns {sorted(missing)}. Path: {fs}")
        if df.empty:
            raise ValueError(f"Filtered dataset is empty ({fs}).")
        gmap = (df[["title", "genres"]].drop_duplicates("title")
                .set_index("title")["genres"].to_dict()) if "genres" in df.columns else {}
        try:
            umm = df.pivot_table(index="userId", columns="title", values="rating", aggfunc="mean").fillna(0)
        except (KeyError, ValueError) as e:
            raise ValueError(f"Could not build user–movie matrix: {e}") from e
        return df, df[["title"]].drop_duplicates(), umm, gmap

    mp, rp = _ensure_data_downloaded()
    movies  = _read_csv_checked(mp, "movies.csv")
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
        umm  = df.pivot_table(index="userId", columns="title", values="rating", aggfunc="mean").fillna(0)
    except (KeyError, ValueError) as e:
        raise ValueError(f"Could not build matrix: {e}") from e
    return df, movies, umm, gmap


# ─────────────────────────────────────────────────────────────────
#  Minimal recommendation placeholder (to be replaced with real logic)
# ─────────────────────────────────────────────────────────────────
def recommend_for_user(username: str, df: pd.DataFrame, movies_df: pd.DataFrame, umm: pd.DataFrame, gmap: Dict[str, str], top_k: int = 6):
    """
    Very simple popularity-based fallback recommendations:
    - If user has watch history, recommend similar genres from that history.
    - Otherwise recommend top-rated/popular movies from the filtered dataset.
    """
    if username == "guest":
        # top-rated by average rating
        top = df.groupby("title")["rating"].mean().sort_values(ascending=False).head(top_k).index.tolist()
        return top
    history = watch_history_list(username, limit=50)
    if history:
        # find genres from history and recommend other movies that share those genres
        seen = set(history)
        genres = []
        for t in history:
            g = gmap.get(t)
            if g:
                genres.extend(g.split("|"))
        if genres:
            genres = set([x.strip() for x in genres if x.strip()])
            candidates = []
            for title, g in gmap.items():
                if title in seen:
                    continue
                if any(gg in g for gg in genres):
                    candidates.append(title)
            # fallback to popularity if not enough
            if len(candidates) >= top_k:
                return candidates[:top_k]
    # final fallback: most popular titles in df
    pop = df["title"].value_counts().head(top_k).index.tolist()
    return pop


# ─────────────────────────────────────────────────────────────────
#  App main
# ─────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="RecoMind", layout="centered", initial_sidebar_state="collapsed")
    st.markdown("<meta name='viewport' content='width=device-width, initial-scale=1'>", unsafe_allow_html=True)

    logged = render_auth_screen()
    if not logged:
        # render_auth_screen handles rerun on login; if not logged, stop here
        return

    username = st.session_state.get("username", "guest")
    st.markdown(DARK_CARD_CSS, unsafe_allow_html=True)

    st.header(f"Welcome, {username}")
    st.write("Personalized emotional movie recommendations powered by simple heuristics (demo).")

    # Controls
    with st.sidebar:
        st.markdown("## Filters")
        top_n_movies = st.slider("Top N movies (popularity filter)", 100, 2000, 500, step=100)
        top_n_users = st.slider("Top N users (activity filter)", 100, 2000, 500, step=100)
        if st.button("Reload data"):
            # clear cache and reload
            load_and_prepare_data.clear()
            st.experimental_rerun()

    # Load data (cached)
    try:
        df, movies_df, umm, gmap = load_and_prepare_data(top_n_movies=top_n_movies, top_n_users=top_n_users)
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        st.info("Place MovieLens CSVs under Data/MovieLens/ or ensure network access for first-time download.")
        return

    # Show a few recommendations
    recs = recommend_for_user(username, df, movies_df, umm, gmap, top_k=6)
    st.subheader("Recommended for you")
    cols = st.columns(3)
    for i, title in enumerate(recs):
        with cols[i % 3]:
            st.markdown(f"**{title}**")
            synopsis = MOVIE_SYNOPSES.get(title, MOVIE_SYNOPSES["default"])
            st.caption(synopsis)
            if st.button(f"Mark watched: {title}", key=f"watched_{i}"):
                watch_history_push(username, title)
                st.success(f"Added {title} to your watch history.")
                st.experimental_rerun()

    st.markdown("---")
    st.subheader("Your recent watch history")
    hist = watch_history_list(username)
    if hist:
        for t in hist:
            st.write(f"- {t}")
    else:
        st.info("No watch history yet. Use the recommendations above or try the Guest Demo.")

    st.markdown("<div style='margin-top:2rem;color:#64748b;font-size:0.9rem;'>RecoMind demo — not for production use.</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
