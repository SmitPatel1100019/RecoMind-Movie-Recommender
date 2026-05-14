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

_POSTER_FALLBACK_HTML = (
    "<div style='background:#0d1117;border:1px solid #1f2937;border-radius:8px;"
    "height:200px;display:flex;align-items:center;justify-content:center;"
    "color:#374151;font-size:2.5rem;'>🎬</div>"
)

MOVIELENS_MOVIE_COLS = frozenset({"movieId", "title", "genres"})
MOVIELENS_RATING_COLS = frozenset({"userId", "movieId", "rating"})

USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_]{3,32}$")
COLD_START_GENRE_OPTIONS = sorted(GENRE_SENTIMENT_MAP.keys())

# ─────────────────────────────────────────────────────────────────
#  CSS — Dark cinematic theme with DM Sans + Space Mono
# ─────────────────────────────────────────────────────────────────
DARK_CARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

[data-testid="stImage"] img {
  transition: transform 0.2s ease;
}
[data-testid="stImage"] img:hover {
  transform: scale(1.03);
}

.top-nav-stats {
  display: flex; justify-content: center; align-items: center;
  gap: 1.75rem; flex-wrap: wrap;
  padding: .65rem 1.1rem; margin: 0 0 .85rem;
  background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
  border: 1px solid #1f2937; border-radius: 10px;
  font-size: .82rem; color: #94a3b8;
}
.top-nav-stats strong { color: #e2e8f0; font-weight: 700; }
.top-nav-stats span { white-space: nowrap; }

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
  background: linear-gradient(145deg, #0f172a, #111827);
  border: 1px solid #1f2937;
  border-radius: 14px;
  padding: 1rem 1.2rem;
  margin-bottom: .8rem;
  transition: all .25s ease;
  overflow: hidden;
  position: relative;
}
.rec-card:hover {
  transform: translateY(-3px);
  border-color: #374151;
  box-shadow: 0 10px 24px rgba(0,0,0,.35);
}
.rec-card::before {
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  height: 3px;
  width: 100%;
  background: linear-gradient(90deg, #ef4444, #8b5cf6, #06b6d4);
}
.rec-title {
  font-size: 1.08rem;
  font-weight: 700;
  color: #f8fafc;
  margin-bottom: .35rem;
  letter-spacing: -.02em;
}
.rec-meta  { font-size: .81rem; color: #6b7280; }
.rec-score {
  font-family: 'Space Mono', monospace; font-size: .76rem; color: #10b981;
  font-weight: 700; background: #052e16; border-radius: 4px;
  padding: 1px 7px; display: inline-block; margin-top: .3rem;
}
.rec-synopsis { font-size: .81rem; color: #6b7280; margin-top: .45rem; line-height: 1.55; }
.rec-explain {
  font-size: .78rem; color: #94a3b8; margin-top: .55rem; line-height: 1.55;
  border-left: 3px solid #4f46e5; padding: .45rem .65rem; background: #0c1220;
  border-radius: 0 8px 8px 0;
}

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
.section-divider { border: none; border-top: 1px solid #1f2937; margin: .8rem 0; }

.stButton > button {
  border-radius: 10px !important;
  font-weight: 600 !important;
  transition: all .2s ease !important;
}
.stButton > button:hover {
  transform: translateY(-1px);
}

.helper-msg {
  background: #0c1a35; border: 1px dashed #1e3a5f; border-radius: 10px;
  padding: 1.2rem 1.5rem; color: #3b82f6; font-size: .9rem;
  text-align: center; margin: 1rem 0;
}

.empty-cinema-banner {
  border-radius: 14px;
  padding: 2.2rem 1.5rem 2rem;
  margin: 0 0 1rem;
  text-align: center;
  background: linear-gradient(145deg, #0a0f1a 0%, #151528 40%, #0f172a 100%);
  border: 1px solid #1e293b;
  position: relative;
  overflow: hidden;
}
.empty-cinema-banner::before {
  content: "";
  position: absolute;
  inset: 0;
  background: radial-gradient(ellipse 80% 50% at 50% 0%, rgba(139, 92, 246, .12), transparent 55%);
  pointer-events: none;
}
.empty-cinema-inner { position: relative; z-index: 1; }
.empty-cinema-kicker {
  font-family: 'Space Mono', monospace;
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: #64748b;
  margin-bottom: .5rem;
}
.empty-cinema-title {
  font-size: 1.35rem;
  font-weight: 700;
  color: #e2e8f0;
  margin-bottom: .35rem;
}
.empty-cinema-sub {
  font-size: .88rem;
  color: #64748b;
  max-width: 28rem;
  margin: 0 auto;
  line-height: 1.5;
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
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_watch_user_time ON watch_history(username, watched_at)"
        )
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
    """Return an error string if password is too weak, else None."""
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
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
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
        return False, "Account data is invalid — please sign up again or use another account."
    if ok:
        return True, "Login successful."
    return False, "Incorrect password."


def watch_history_list(username: str, limit: int = 40) -> List[str]:
    if username == "guest":
        return []
    try:
        with _db_connect() as conn:
            rows = conn.execute(
                """
                SELECT movie_title FROM watch_history
                WHERE username = ? ORDER BY watched_at DESC LIMIT ?
                """,
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
        pass


def rec_feedback_push(username: str, movie_title: str, anchor_title: str, sentiment: str) -> None:
    if username == "guest" or not movie_title or not sentiment:
        return
    try:
        with _db_connect() as conn:
            conn.execute(
                """
                INSERT INTO rec_feedback (username, movie_title, anchor_title, sentiment)
                VALUES (?, ?, ?, ?)
                """,
                (username.strip().lower(), movie_title, anchor_title or "", sentiment),
            )
    except sqlite3.Error:
        pass


def render_auth_screen() -> bool:
    """Returns True if user is logged in. Shows login/signup/guest UI otherwise."""
    _init_auth_db()

    if st.session_state.get("logged_in"):
        return True

    st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }

    .main > div {
        padding-top: 0rem;
    }

    .auth-container {
        max-width: 520px;
        margin: 80px auto 0 auto !important;
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

    .auth-container .stCaption {
        text-align: center;
        display: block;
        margin: -0.75rem 0 1.25rem;
        color: #64748b !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Start centered container
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)

    st.markdown(
        "<div class='auth-title'>🎬 RecoMind</div>",
        unsafe_allow_html=True
    )
    st.caption("AI-powered emotional movie recommendation system")

    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    with tab_login:
        uname = st.text_input("Username", key="login_user")
        st.caption("Registered usernames: letters, digits, underscore · 3–32 characters.")
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
        st.caption("3–32 characters: letters, digits, or underscore only (example: smit_patel).")

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

        st.markdown("**Cold start profile (optional)**")
        st.caption("Helps mitigate the new-user cold-start problem before you have watch history.")
        fav_gen = st.multiselect(
            "Favourite genres", COLD_START_GENRE_OPTIONS, key="signup_fav_gen"
        )
        fav_movie = st.text_input(
            "A favourite movie (partial title is OK)", key="signup_fav_movie", max_chars=180
        )
        mood_pre = st.selectbox(
            "Preferred mood vibe",
            ["—"] + list(MOOD_GENRE_MAP.keys()),
            key="signup_mood",
        )

        if st.button("Create Account", use_container_width=True):
            ok, msg = _signup(
                new_user,
                new_pwd,
                confirm_pwd,
                fav_gen or None,
                fav_movie,
                mood_pre,
            )

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
    try:
        with st.spinner("Downloading MovieLens dataset…"):
            urllib.request.urlretrieve(ML_LATEST_SMALL_URL, zp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"MovieLens download failed (HTTP {e.code}). Check your connection.") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            "MovieLens download failed (no network or DNS error). "
            "Connect to the internet or place movies.csv and ratings.csv under Data/MovieLens/."
        ) from e
    except OSError as e:
        raise RuntimeError(f"MovieLens download failed (disk error): {e}") from e

    try:
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(DATA_DIR)
    except zipfile.BadZipFile as e:
        raise RuntimeError(
            "Downloaded MovieLens archive is corrupted or not a zip file. Delete ml-latest-small.zip and retry."
        ) from e
    except OSError as e:
        raise RuntimeError(f"Could not extract MovieLens archive: {e}") from e

    if not (os.path.exists(em) and os.path.exists(er)):
        raise FileNotFoundError("Download finished but movies.csv / ratings.csv were not found in the archive.")
    return em, er


def _read_csv_checked(path: str, label: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except (ParserError, EmptyDataError) as e:
        raise ValueError(f"{label}: CSV is empty or could not be parsed ({path}): {e}") from e
    except (OSError, UnicodeDecodeError) as e:
        raise ValueError(f"{label}: could not read file ({path}): {e}") from e
    return df


@st.cache_data(show_spinner=False)
def load_and_prepare_data(top_n_movies: int, top_n_users: int):
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
            raise ValueError(
                f"Filtered CSV is missing required columns {sorted(missing)}. "
                f"Found columns: {list(df.columns)}. Path: {fs}"
            )
        if df.empty:
            raise ValueError(f"Filtered dataset is empty ({fs}).")
        gmap = (df[["title", "genres"]].drop_duplicates("title")
                .set_index("title")["genres"].to_dict()) if "genres" in df.columns else {}
        try:
            umm = df.pivot_table(index="userId", columns="title",
                                 values="rating", aggfunc="mean").fillna(0)
        except (KeyError, ValueError) as e:
            raise ValueError(f"Could not build user–movie matrix from filtered data: {e}") from e
        return df, df[["title"]].drop_duplicates(), umm, gmap

    mp, rp = _ensure_data_downloaded()
    movies = _read_csv_checked(mp, "movies.csv")
    ratings = _read_csv_checked(rp, "ratings.csv")
    m_miss = MOVIELENS_MOVIE_COLS - set(movies.columns)
    r_miss = MOVIELENS_RATING_COLS - set(ratings.columns)
    if m_miss:
        raise ValueError(
            f"movies.csv missing columns {sorted(m_miss)}. Found: {list(movies.columns)}"
        )
    if r_miss:
        raise ValueError(
            f"ratings.csv missing columns {sorted(r_miss)}. Found: {list(ratings.columns)}"
        )

    try:
        df = pd.merge(ratings, movies, on="movieId", how="inner")
    except KeyError as e:
        raise ValueError(f"Could not merge ratings and movies on movieId: {e}") from e

    df.dropna(inplace=True)
    if df.empty:
        raise ValueError("Merged MovieLens data is empty after removing null rows.")

    try:
        top_titles = df["title"].value_counts().head(top_n_movies).index
        top_users = df["userId"].value_counts().head(top_n_users).index
        df = df[df["title"].isin(top_titles)]
        df = df[df["userId"].isin(top_users)]
    except (KeyError, TypeError) as e:
        raise ValueError(f"Could not filter ratings by popularity: {e}") from e

    if df.empty:
        raise ValueError(
            "No ratings left after applying Top Movies / Active Users filters. "
            "Lower those sliders in the sidebar and rebuild."
        )

    try:
        gmap = movies[["title", "genres"]].drop_duplicates("title").set_index("title")["genres"].to_dict()
        umm = df.pivot_table(index="userId", columns="title",
                             values="rating", aggfunc="mean").fillna(0)
    except (KeyError, ValueError) as e:
        raise ValueError(f"Could not build genre map or user–movie matrix: {e}") from e

    return df, movies, umm, gmap


@st.cache_data(show_spinner=False)
def compute_avg_ratings(top_n_movies: int, top_n_users: int) -> Dict[str, float]:
    df, _, _, _ = load_and_prepare_data(top_n_movies, top_n_users)
    if df.empty or "rating" not in df.columns or "title" not in df.columns:
        return {}
    try:
        return df.groupby("title")["rating"].mean().round(2).to_dict()
    except (TypeError, KeyError):
        return {}


@st.cache_resource(show_spinner=True)
def build_recommender(top_n_movies: int, top_n_users: int):
    _, _, umm, gmap = load_and_prepare_data(top_n_movies, top_n_users)

    if umm.empty or umm.shape[1] == 0:
        raise ValueError(
            "The training matrix has no movie columns. "
            "Increase Top Movies / Active Users in the sidebar or verify your CSV."
        )

    try:
        X = umm.to_numpy(dtype=np.float32)
        norms = np.linalg.norm(X, axis=0)
        norms[norms == 0] = 1e-8
        Xn = X / norms
        sim = Xn.T @ Xn
    except MemoryError as e:
        raise RuntimeError(f"Not enough memory to build the similarity matrix: {e}") from e

    titles = umm.columns
    t2i = {t: i for i, t in enumerate(titles)}
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


def primary_genre_label(gs: str) -> str:
    s = genres_set(gs)
    return next(iter(sorted(s)), "Other")


@st.cache_data(show_spinner=False)
def sorted_movie_catalog(movie_titles_tuple: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(sorted(movie_titles_tuple))


def explain_recommendation(
    anchor: str,
    rec_title: str,
    gmap: Dict[str, str],
    avg_ratings: Dict[str, float],
    mood_genres: Set[str],
) -> str:
    if not anchor:
        return (
            "<strong>Why this title?</strong> Selected using hybrid collaborative filtering "
            "and dataset ratings (mood-weighted when a mood profile is active)."
        )
    ag = gmap.get(anchor, "")
    rg = gmap.get(rec_title, "")
    aset, rset = genres_set(ag), genres_set(rg)
    shared = sorted(aset & rset)
    ca = html.escape(clean_title(anchor))
    cr = html.escape(clean_title(rec_title))
    parts: List[str] = []
    if shared:
        sg = html.escape(", ".join(shared[:6]) + ("…" if len(shared) > 6 else ""))
        parts.append(
            f"<strong>Why this title?</strong> {ca} and {cr} share catalogue genres ({sg}), "
            f"so audience taste patterns align in the latent factor space."
        )
    r_sim = avg_ratings.get(rec_title)
    r_an = avg_ratings.get(anchor)
    if r_sim is not None and r_an is not None:
        parts.append(
            f"Users who rated the anchor context around ⭐{r_an} also tended to rate "
            f"<em>{cr}</em> near ⭐{r_sim} in this MovieLens slice."
        )
    if mood_genres and (rset & mood_genres):
        mg = html.escape(", ".join(sorted(rset & mood_genres)[:5]))
        parts.append(f"Overlaps your active mood genres: {mg}.")
    if not parts:
        parts.append(
            f"<strong>Why this title?</strong> Strong collaborative match from user–item ratings: "
            f"viewers in the embedding neighbourhood of <em>{ca}</em> frequently co-rated <em>{cr}</em>."
        )
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────
#  TMDB
# ─────────────────────────────────────────────────────────────────
def _tmdb_key() -> Optional[str]:
    k = os.environ.get("TMDB_API_KEY", "").strip()
    if k:
        return k
    try:
        return str(st.secrets["TMDB_API_KEY"]).strip() or None
    except (KeyError, TypeError, AttributeError):
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def _tmdb_result(api_key: str, ct: str, yr: str) -> Optional[dict]:
    if not api_key or not ct:
        return None
    params: Dict[str, str] = {"api_key": api_key, "query": ct}
    if yr and yr.isdigit():
        params["year"] = yr
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params=params,
            timeout=12,
        )
    except requests.RequestException:
        return None

    if r.status_code != 200:
        return None

    try:
        data = r.json()
    except ValueError:
        return None

    if not isinstance(data, dict):
        return None
    raw = data.get("results")
    if not isinstance(raw, list) or not raw:
        return None
    first = raw[0]
    if not isinstance(first, dict):
        return None
    return first


def get_poster(api_key: Optional[str], full: str) -> Optional[str]:
    if not api_key:
        return None
    y = extract_year(full)
    ct = clean_title(full)
    try:
        res = _tmdb_result(api_key, ct, y if y != "—" else "")
    except (TypeError, AttributeError):
        return None
    if not res:
        return None
    p = res.get("poster_path")
    if not isinstance(p, str) or not p.strip():
        return None
    return f"https://image.tmdb.org/t/p/w342{p.strip()}"


def get_synopsis(api_key: Optional[str], full: str) -> str:
    y = extract_year(full)
    ct = clean_title(full)
    if api_key:
        try:
            res = _tmdb_result(api_key, ct, y if y != "—" else "")
        except (TypeError, AttributeError):
            res = None
        if res and isinstance(res, dict):
            ov = (res.get("overview") or "").strip()
            if ov:
                return ov
    return MOVIE_SYNOPSES.get(full, MOVIE_SYNOPSES["default"])


def get_tmdb_movie_id(api_key: Optional[str], full: str) -> Optional[int]:
    if not api_key:
        return None
    y = extract_year(full)
    ct = clean_title(full)
    try:
        res = _tmdb_result(api_key, ct, y if y != "—" else "")
    except (TypeError, AttributeError):
        return None
    if not res or not isinstance(res, dict):
        return None
    mid = res.get("id")
    if isinstance(mid, int):
        return mid
    if isinstance(mid, str) and str(mid).isdigit():
        return int(mid)
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def _tmdb_fetch_review_texts(api_key: str, movie_id: int) -> Tuple[str, ...]:
    if not api_key or movie_id <= 0:
        return tuple()
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/reviews",
            params={"api_key": api_key},
            timeout=14,
        )
    except requests.RequestException:
        return tuple()
    if r.status_code != 200:
        return tuple()
    try:
        data = r.json()
    except ValueError:
        return tuple()
    if not isinstance(data, dict):
        return tuple()
    raw = data.get("results")
    if not isinstance(raw, list):
        return tuple()
    out: List[str] = []
    for item in raw[:20]:
        if not isinstance(item, dict):
            continue
        c = (item.get("content") or "").strip()
        if c:
            out.append(c[:1200])
    return tuple(out)


def show_poster(api_key: Optional[str], full: str) -> None:
    if not api_key:
        return
    pu = get_poster(api_key, full)
    if not pu:
        return
    try:
        st.image(pu, use_container_width=True)
    except Exception:
        st.markdown(_POSTER_FALLBACK_HTML, unsafe_allow_html=True)


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
    max_scan: int = 4000,
) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["movie", "score", "genres", "mood_match"])
    try:
        if movie_name not in t2i:
            return empty

        idx = t2i[movie_name]
        n = len(movie_titles)
        sim_scores = np.clip(sim_matrix[idx].copy().astype(np.float32), 0.0, 1.0)

        max_r = max(avg_ratings.values()) if avg_ratings else 5.0
        if max_r == 0:
            max_r = 5.0

        rating_norm = np.array(
            [avg_ratings.get(str(movie_titles[i]), 0.0) / max_r for i in range(n)],
            dtype=np.float32,
        )

        if mood_genres:
            mg_den = float(max(1, len(mood_genres)))
            mood_arr = np.zeros(n, dtype=np.float32)
            for i in range(n):
                inter = len(genres_set(gmap.get(movie_titles[i], "")) & mood_genres)
                mood_arr[i] = min(1.0, inter / mg_den)
            hybrid = (sim_scores * 0.55 + rating_norm * 0.25 + mood_arr * 0.20).astype(np.float32)
        else:
            hybrid = (sim_scores * 0.70 + rating_norm * 0.30).astype(np.float32)

        order = np.argsort(-hybrid).tolist()
        cand: List[int] = []
        for i in order:
            if i == idx:
                continue
            if excluded_match(gmap.get(movie_titles[i], ""), excl):
                continue
            cand.append(i)
            if len(cand) >= max_scan:
                break

        picked: List[int] = []
        genre_count: Dict[str, int] = {}
        picked_set: Set[int] = set()

        for i in cand:
            if len(picked) >= top_k:
                break
            pg = primary_genre_label(gmap.get(movie_titles[i], ""))
            if genre_count.get(pg, 0) >= 2:
                continue
            picked.append(i)
            picked_set.add(i)
            genre_count[pg] = genre_count.get(pg, 0) + 1

        if len(picked) < top_k:
            for i in cand:
                if len(picked) >= top_k:
                    break
                if i in picked_set:
                    continue
                picked.append(i)
                picked_set.add(i)

        if not picked:
            return empty

        df = pd.DataFrame({"movie": movie_titles[picked], "score": hybrid[picked]})
        df["genres"] = df["movie"].map(lambda t: gmap.get(t, ""))
        df["mood_match"] = df["genres"].apply(
            lambda g: bool(genres_set(g) & mood_genres) if mood_genres else False
        )
        return df
    except (IndexError, KeyError, TypeError, ValueError):
        return empty


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
            f'<span class="why-tag tag-anchor">🎬 Fans of {html.escape(clean_title(anchor)[:22])} also watched</span>'
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
            try:
                st.image(pu, use_container_width=True)
            except Exception:
                st.markdown(_POSTER_FALLBACK_HTML, unsafe_allow_html=True)
        else:
            st.markdown(_POSTER_FALLBACK_HTML, unsafe_allow_html=True)
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
    gmap: Dict[str, str],
    anchor: str = "",
    from_history: bool = False,
    mood_genres: Optional[Set[str]] = None,
    feedback_key_prefix: str = "rec",
) -> None:
    mood_genres = mood_genres or set()
    if recs.empty:
        st.warning("No recommendations found for the current filters.")
        return

    st.markdown("""
<div style='display:flex;align-items:center;gap:10px;margin-bottom:.6rem;flex-wrap:wrap;'>
<h2 style='margin:0;font-size:1.35rem;'>🎯 Top Recommendations</h2>
<span style='font-size:.8rem;color:#64748b;'>
AI-powered personalised matches
</span>
</div>
""", unsafe_allow_html=True)

    uname = st.session_state.get("username", "guest")

    for i, (_, row) in enumerate(recs.iterrows()):
        title = str(row["movie"])
        score = float(row["score"])
        match_pct = min(99, int(score * 100))
        graw = str(row.get("genres", ""))
        gd = graw.replace("|", "  ·  ") if graw else "—"
        year = extract_year(title)
        ct = clean_title(title)
        rating = avg_ratings.get(title)
        rstr = f"⭐ {rating}" if rating else "—"
        mmatch = bool(row.get("mood_match", False))
        syn = get_synopsis(api_key, title)
        why = build_why_tags(title, score, mmatch, avg_ratings, anchor, from_history)
        expl = explain_recommendation(anchor, title, gmap, avg_ratings, mood_genres)

        ct_safe = html.escape(ct)
        gd_safe = html.escape(gd)
        syn_safe = html.escape(syn)

        pc, ic = st.columns([1, 3])
        with pc:
            show_poster(api_key, title)
        with ic:
            st.markdown(
                f"<div class='rec-card'>"
                f"<div class='rec-title'>{ct_safe} <span style='color:#374151'>({html.escape(year)})</span></div>"
                f"<div class='rec-meta'>{gd_safe}&nbsp;·&nbsp;{rstr}</div>"
                f"<span class='rec-score'>🎯 Relevance Score: {match_pct}%</span>"
                f"<div class='rec-synopsis'>{syn_safe}</div>"
                f"{why}"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"<div class='rec-explain'>{expl}</div>", unsafe_allow_html=True)
            if uname != "guest":
                b1, b2, b3 = st.columns(3)
                with b1:
                    if st.button("👍 Helpful", key=f"{feedback_key_prefix}_fb_help_{i}", use_container_width=True):
                        rec_feedback_push(uname, title, anchor, "helpful")
                with b2:
                    if st.button("👎 Not for me", key=f"{feedback_key_prefix}_fb_skip_{i}", use_container_width=True):
                        rec_feedback_push(uname, title, anchor, "not_interested")
                with b3:
                    if st.button("❤️ Loved it", key=f"{feedback_key_prefix}_fb_love_{i}", use_container_width=True):
                        rec_feedback_push(uname, title, anchor, "loved")
            else:
                st.caption("Sign in to persist 👍 / 👎 / ❤️ feedback in SQLite.")


# ─────────────────────────────────────────────────────────────────
#  Sentiment analysis section
# ─────────────────────────────────────────────────────────────────
def render_sentiment_section(selected_movie: str, tmdb_key: Optional[str] = None) -> None:
    import plotly.express as px
    from textblob import TextBlob

    st.markdown("<div class='section-hdr'>🎭 Audience Sentiment Analysis</div>", unsafe_allow_html=True)

    reviews: List[str] = list(SAMPLE_REVIEWS.get(selected_movie, SAMPLE_REVIEWS["default"]))
    use_live = False
    if tmdb_key:
        mid = get_tmdb_movie_id(tmdb_key, selected_movie)
        if mid is not None:
            ext = list(_tmdb_fetch_review_texts(tmdb_key, mid))
            if len(ext) >= 2:
                reviews = ext
                use_live = True

    if use_live:
        st.caption(
            "Live TMDB review excerpts (where available) · TextBlob polarity for explainable sentiment."
        )
    else:
        st.caption(
            "NLP sentiment scoring via TextBlob. "
            "Prototype sample reviews when TMDB excerpts are unavailable — "
            "future scope: larger corpora or transformer models."
        )

    scored = []
    try:
        for r in reviews:
            pol = TextBlob(str(r)).sentiment.polarity
            label = "Positive 😊" if pol > 0.05 else ("Negative 😟" if pol < -0.05 else "Neutral 😐")
            scored.append({"Review": r, "Sentiment": label, "Polarity": round(pol, 3)})
    except Exception as e:
        st.warning(f"Sentiment scoring failed ({type(e).__name__}: {e}). Showing neutral placeholders.")
        scored = [
            {"Review": r, "Sentiment": "Neutral 😐", "Polarity": 0.0}
            for r in reviews
        ]

    scored_df = pd.DataFrame(scored)
    st.markdown("**📋 Sample Reviews**")
    st.table(scored_df)

    sc = scored_df["Sentiment"].value_counts().reset_index()
    sc.columns = ["Sentiment", "Count"]
    try:
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
    except Exception as e:
        st.warning(f"Could not render sentiment chart ({type(e).__name__}: {e}).")

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
        try:
            up = TextBlob(user_review).sentiment.polarity
            ul = "Positive 😊" if up > 0.05 else ("Negative 😟" if up < -0.05 else "Neutral 😐")
            st.success(f"**Your review sentiment:** {ul}  (Polarity: {up:.3f})")
        except Exception as e:
            st.warning(f"Could not analyse that review ({type(e).__name__}: {e}).")


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
            "👤 Guest mode — sign in to persist watch history, feedback, and profile in SQLite."
            "</div>",
            unsafe_allow_html=True,
        )

    last = st.session_state["last_built_params"]

    # ── Trending strip ──────────────────────────────────────────
    df_ratings, _, _, _ = load_and_prepare_data(last["top_n_movies"], last["top_n_users"])
    top_counts = df_ratings["title"].value_counts().head(24)

    trending_rows: List[dict] = []
    for t in top_counts.index:
        if excluded_match(gmap.get(t, ""), excluded_frozen):
            continue
        trending_rows.append({
            "Movie": clean_title(t),
            "Year": extract_year(t),
            "Avg Rating": round(float(avg_ratings.get(t, 0.0)), 2),
            "# Ratings": int(top_counts[t]),
        })
        if len(trending_rows) >= 8:
            break
    if trending_rows:
        st.markdown("<div class='section-hdr'>🔥 Trending Now</div>", unsafe_allow_html=True)
        trending_df = pd.DataFrame(trending_rows)
        st.dataframe(trending_df, use_container_width=True, hide_index=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Unified Control Panel ────────────────────────────────────
    st.markdown(
        "<div class='control-panel'>"
        "<div class='control-panel-title'>🤖 RecoMind — Unified Engine</div>",
        unsafe_allow_html=True,
    )

    base_titles = sorted_movie_catalog(tuple(movie_titles.tolist()))
    all_sorted = list(base_titles)
    if excluded_frozen:
        all_sorted = [t for t in all_sorted if not excluded_match(gmap.get(t, ""), excluded_frozen)]

    col_movie, col_genre = st.columns([3, 1])
    with col_genre:
        genre_filter = st.selectbox("🎭 Genre", ["All"] + genre_options, index=0, key="uni_genre")
    if genre_filter != "All":
        all_sorted = [t for t in all_sorted if genre_filter in genres_set(gmap.get(t, ""))]
    with col_movie:
        selected_movie = st.selectbox(
            "Movie",
            options=all_sorted,
            index=None,
            placeholder="Type movie name...",
            key="movie_searchbox",
            help="Type to filter titles — same list powers recommendations below.",
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
            f"🎭 Mood active: {selected_mood} — genres weighted at 20% in the hybrid score.</div>",
            unsafe_allow_html=True,
        )
    elif has_excl:
        st.markdown(
            f"<div class='filter-banner banner-exclude'>"
            f"🚫 Wellbeing filter: excluding {', '.join(sorted(excluded_frozen))}</div>",
            unsafe_allow_html=True,
        )

    # Empty state (single picker: selectbox only; mood optional)
    if not selected_movie and not has_mood:
        st.markdown(
            "<div class='empty-cinema-banner'>"
            "<div class='empty-cinema-inner'>"
            "<div class='empty-cinema-kicker'>RecoMind · Ready</div>"
            "<div class='empty-cinema-title'>🎬 Choose a title to begin</div>"
            "<div class='empty-cinema-sub'>"
            "Use the field above to search the catalogue, or activate a mood for "
            "instant mood-aware picks — recommendations update automatically."
            "</div></div></div>",
            unsafe_allow_html=True,
        )
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

    # Skeleton + spinner while hybrid engine runs
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    sk = st.empty()
    with st.spinner("Generating recommendations…"):
        sk.markdown("".join([SKELETON_HTML] * 3), unsafe_allow_html=True)
        recs = recommend(
            sim_matrix=sim_matrix, movie_titles=movie_titles, gmap=gmap, t2i=t2i,
            movie_name=anchor, top_k=top_k, avg_ratings=avg_ratings,
            excl=excluded_frozen, mood_genres=mood_genres,
        )
    _rec_sig = (
        str(selected_movie or ""),
        str(selected_mood or ""),
        str(anchor or ""),
        str(genre_filter),
        int(top_k),
        tuple(sorted(excluded_frozen)),
    )
    if st.session_state.get("_rec_gen_sig") != _rec_sig:
        st.session_state["rec_gens"] = st.session_state.get("rec_gens", 0) + 1
        st.session_state["_rec_gen_sig"] = _rec_sig

    uname = st.session_state.get("username", "guest")
    if uname != "guest" and anchor and anchor in t2i:
        watch_history_push(uname, anchor)
    elif uname == "guest" and anchor and anchor in t2i:
        gh = st.session_state.get("watch_history", [])
        gh = [m for m in gh if m != anchor]
        gh.insert(0, anchor)
        st.session_state["watch_history"] = gh[:5]

    sk.empty()

    if excluded_frozen:
        st.info(f"🚫 Wellbeing filter active — {', '.join(sorted(excluded_frozen))} excluded from all results.")

    st.caption(
        "**Hybrid ranking:** with a mood selected → 55% collaborative similarity + 25% audience rating + "
        "**20% mood–genre alignment**. Without mood → 70% / 30%. "
        "Diversity cap: at most two picks per primary genre before filling remaining slots."
    )
    render_rec_cards(
        recs, avg_ratings, tmdb_key, top_k, gmap,
        anchor=anchor or "", mood_genres=mood_genres,
    )

    # Sentiment section (only when a movie is explicitly selected)
    if selected_movie:
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        render_sentiment_section(selected_movie, tmdb_key)

    if uname == "guest":
        hist = st.session_state.get("watch_history", [])
    else:
        hist = watch_history_list(uname)
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
                hr = recommend(
                    sim_matrix, movie_titles, gmap, t2i, ht, 20,
                    avg_ratings, excluded_frozen, mood_genres=mood_genres,
                )
                all_recs.append(hr)
        if all_recs:
            combined = pd.concat(all_recs).drop_duplicates("movie")
            combined = combined[~combined["movie"].isin(hist)]
            combined = (combined.groupby("movie", as_index=False)
                        .agg({"score": "mean", "genres": "first", "mood_match": "any"})
                        .sort_values("score", ascending=False).head(8).reset_index(drop=True))
            render_rec_cards(
                combined, avg_ratings, tmdb_key, 8, gmap,
                from_history=True, mood_genres=mood_genres,
                feedback_key_prefix="hist",
            )


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

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    with st.expander("Offline evaluation snapshot (MAE — report-friendly)"):
        st.markdown(
            "Hold-out **10%** of ratings (capped at 8k rows). Compare **global-mean** vs "
            "**per-movie mean** predictors trained only on the remaining 90% — common baselines "
            "for recommender write-ups."
        )
        if len(df_dash) < 200:
            st.info("Not enough ratings for a stable hold-out in this slice.")
        else:
            n_hold = min(8000, max(500, len(df_dash) // 10))
            hold = df_dash.sample(n=n_hold, random_state=42)
            train = df_dash.drop(hold.index)
            g_mean = float(train["rating"].mean())
            m_mean = train.groupby("title")["rating"].mean()
            h2 = hold.merge(m_mean.rename("pred_movie"), on="title", how="left")
            h2["pred_movie"] = h2["pred_movie"].fillna(g_mean)
            mae_movie = float((h2["rating"] - h2["pred_movie"]).abs().mean())
            mae_glob = float((h2["rating"] - g_mean).abs().mean())
            c_m, c_g = st.columns(2)
            c_m.metric("MAE per-movie mean", f"{mae_movie:.3f}")
            c_g.metric("MAE global mean", f"{mae_glob:.3f}")
            st.caption(
                "Precision@K / Recall@K need relevance judgements; document protocol in the report."
            )


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="RecoMind", page_icon="🎬", layout="wide")
    st.markdown(DARK_CARD_CSS, unsafe_allow_html=True)

    if not render_auth_screen():
        st.stop()

    _init_auth_db()

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

    try:
        with st.spinner("⚙️ Building recommendation model…"):
            sim_matrix, gmap, movie_titles, t2i = build_recommender(
                last["top_n_movies"], last["top_n_users"]
            )
        avg_ratings = compute_avg_ratings(last["top_n_movies"], last["top_n_users"])
        genre_options = collect_genre_options(gmap)
        tmdb_key = _tmdb_key()
        df_nav, _, _, _ = load_and_prepare_data(last["top_n_movies"], last["top_n_users"])
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        st.error(f"**Data or model error:** {e}")
        st.info(
            "Verify CSV files under `Data/MovieLens/`, required columns (userId, title, rating, …), "
            "network for first-time MovieLens download, and sidebar training limits."
        )
        st.stop()
    except Exception as e:
        st.error(f"**Unexpected error while loading the recommender:** {type(e).__name__}: {e}")
        st.stop()

    n_movies_nav = int(df_nav["title"].nunique())
    n_users_nav = int(df_nav["userId"].nunique())
    n_recs_nav = int(st.session_state.get("rec_gens", 0))
    st.markdown(
        f"<div class='top-nav-stats'>"
        f"<span>🎬 Movies <strong>{n_movies_nav:,}</strong></span>"
        f"<span>👥 Users <strong>{n_users_nav:,}</strong></span>"
        f"<span>✨ Recommendations generated <strong>{n_recs_nav:,}</strong></span>"
        f"</div>",
        unsafe_allow_html=True,
    )

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
            st.session_state.pop("rec_gens", None)
            st.session_state.pop("_rec_gen_sig", None)
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
