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

FILTERED_DATA_CSV_LOCAL = os.path.join(DATA_DIR, "filtered_data.csv")
FILTERED_MOVIES_DATA_CSV_LOCAL = os.path.join(DATA_DIR, "filtered_movies_data.csv")

ML_LATEST_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ML_LATEST_SMALL_DIR = os.path.join(DATA_DIR, "ml-latest-small")


def _ensure_data_downloaded() -> Tuple[str, str]:
    if os.path.exists(MOVIES_CSV_LOCAL) and os.path.exists(RATINGS_CSV_LOCAL):
        return MOVIES_CSV_LOCAL, RATINGS_CSV_LOCAL

    os.makedirs(DATA_DIR, exist_ok=True)

    extracted_movies = os.path.join(ML_LATEST_SMALL_DIR, "movies.csv")
    extracted_ratings = os.path.join(ML_LATEST_SMALL_DIR, "ratings.csv")
    if os.path.exists(extracted_movies) and os.path.exists(extracted_ratings):
        return extracted_movies, extracted_ratings

    zip_path = os.path.join(DATA_DIR, "ml-latest-small.zip")
    with st.spinner("Downloading MovieLens (ml-latest-small) ..."):
        urllib.request.urlretrieve(ML_LATEST_SMALL_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    if not (os.path.exists(extracted_movies) and os.path.exists(extracted_ratings)):
        raise FileNotFoundError("MovieLens download completed, but CSVs were not found.")

    return extracted_movies, extracted_ratings


@st.cache_data(show_spinner=False)
def load_and_prepare_data(
    top_n_movies: int,
    top_n_users: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    filtered_source = None
    if os.path.exists(FILTERED_MOVIES_DATA_CSV_LOCAL):
        filtered_source = FILTERED_MOVIES_DATA_CSV_LOCAL
    elif os.path.exists(FILTERED_DATA_CSV_LOCAL):
        filtered_source = FILTERED_DATA_CSV_LOCAL

    if filtered_source is not None:
        df = pd.read_csv(filtered_source)
        if not {"userId", "title", "rating"}.issubset(df.columns):
            raise ValueError(
                "Filtered CSV exists but doesn't have required columns: "
                "expected at least userId, title, rating."
            )

        if "genres" in df.columns:
            genres_map = (
                df[["title", "genres"]]
                .drop_duplicates(subset=["title"])
                .set_index("title")["genres"]
                .to_dict()
            )
        else:
            genres_map = {}

        user_movie_matrix = (
            df.pivot_table(index="userId", columns="title", values="rating", aggfunc="mean")
            .fillna(0)
        )

        movies = df[["title"]].drop_duplicates().rename(columns={"title": "title"})
        return df, movies, user_movie_matrix, genres_map

    movies_path, ratings_path = _ensure_data_downloaded()

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    df = pd.merge(ratings, movies, on="movieId")
    df.dropna(inplace=True)

    top_movies = df["title"].value_counts().head(top_n_movies).index
    df = df[df["title"].isin(top_movies)]

    active_users = df["userId"].value_counts().head(top_n_users).index
    df = df[df["userId"].isin(active_users)]

    genres_map = (
        movies[["title", "genres"]]
        .drop_duplicates(subset=["title"])
        .set_index("title")["genres"]
        .to_dict()
    )

    user_movie_matrix = (
        df.pivot_table(index="userId", columns="title", values="rating", aggfunc="mean")
        .fillna(0)
    )

    return df, movies, user_movie_matrix, genres_map


@st.cache_resource(show_spinner=True)
def build_recommender(
    top_n_movies: int,
    top_n_users: int,
) -> Tuple[np.ndarray, Dict[str, str], pd.Index, Dict[str, int]]:
    _, _, user_movie_matrix, genres_map = load_and_prepare_data(
        top_n_movies=top_n_movies, top_n_users=top_n_users
    )

    X = user_movie_matrix.to_numpy(dtype=np.float32)
    col_norms = np.linalg.norm(X, axis=0)
    col_norms[col_norms == 0] = 1e-8
    Xn = X / col_norms
    sim_matrix = Xn.T @ Xn

    movie_titles = user_movie_matrix.columns
    title_to_index = {title: i for i, title in enumerate(movie_titles)}
    return sim_matrix, genres_map, movie_titles, title_to_index


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
    """Returns a sentiment label based on the dominant genre tone."""
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
    tags = genres_string_to_set(genres_str)
    return bool(tags & set(excluded))


def _tmdb_api_key() -> Optional[str]:
    k = os.environ.get("TMDB_API_KEY", "").strip()
    if k:
        return k
    try:
        return str(st.secrets["TMDB_API_KEY"]).strip() or None
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def tmdb_poster_url(api_key: str, clean_title: str, year_str: str) -> Optional[str]:
    if not api_key or not clean_title:
        return None
    url = "https://api.themoviedb.org/3/search/movie"
    params: Dict[str, str] = {"api_key": api_key, "query": clean_title}
    if year_str and year_str.isdigit():
        params["year"] = year_str
    try:
        r = requests.get(url, params=params, timeout=12)
        if r.status_code != 200:
            return None
        results = (r.json() or {}).get("results") or []
        if not results:
            return None
        path = results[0].get("poster_path")
        if not path:
            return None
        return f"https://image.tmdb.org/t/p/w342{path}"
    except requests.RequestException:
        return None


def show_poster_for_title(api_key: Optional[str], full_title: str, caption: Optional[str] = None) -> None:
    if not api_key:
        return
    y = extract_year_from_title(full_title)
    yq = y if y != "—" else ""
    pu = tmdb_poster_url(api_key, clean_title_for_tmdb(full_title), yq)
    if pu:
        st.image(pu, caption=caption or full_title[:80], use_container_width=True)


def recommend_from_similarity(
    sim_matrix: np.ndarray,
    movie_titles: pd.Index,
    genres_map: Dict[str, str],
    title_to_index: Dict[str, int],
    movie_name: str,
    top_k: int,
    excluded_genres: FrozenSet[str] = frozenset(),
    max_scan: int = 4000,
) -> pd.DataFrame:
    if movie_name not in movie_titles:
        return pd.DataFrame([{"movie": movie_name, "score": 0.0, "genres": genres_map.get(movie_name, "")}])

    movie_index = title_to_index[movie_name]
    scores = sim_matrix[movie_index]

    order = np.argsort(-scores)
    picked_idx: List[int] = []
    scanned = 0
    for i in order.tolist():
        if i == movie_index:
            continue
        scanned += 1
        if scanned > max_scan:
            break
        title_i = movie_titles[i]
        g = genres_map.get(title_i, "")
        if movie_matches_excluded_genres(g, excluded_genres):
            continue
        picked_idx.append(i)
        if len(picked_idx) >= top_k:
            break

    if not picked_idx:
        return pd.DataFrame(columns=["movie", "score", "genres"])

    rec_titles = movie_titles[picked_idx]
    rec_scores = scores[picked_idx]
    rec_df = pd.DataFrame({"movie": rec_titles, "score": rec_scores})
    rec_df["genres"] = rec_df["movie"].map(lambda t: genres_map.get(t, ""))
    return rec_df


def main() -> None:
    st.set_page_config(page_title="RecoMind", page_icon="🎬", layout="wide")
    st.title("RecoMind - Movie Recommendations")
    st.caption("Collaborative filtering using cosine similarity over user ratings.")
    st.info(
        "Search for a movie, optionally **exclude genres** (wellbeing / comfort viewing), then tap **Recommend**. "
        "Suggestions use **similar rating patterns**; posters load from **TMDB** when an API key is set."
    )

    with st.sidebar:
        st.header("Recommendation Settings")

        with st.form("train_form"):
            top_n_movies = st.slider(
                "Top Movies (training filter)",
                min_value=50,
                max_value=300,
                value=100,
                step=10,
            )
            top_n_users = st.slider(
                "Active Users (training filter)",
                min_value=100,
                max_value=2000,
                value=500,
                step=100,
            )
            build_submit = st.form_submit_button("Build/Refresh model")

        top_k = st.slider("Recommendations to show", min_value=5, max_value=30, value=10, step=1)

    if "watch_history" not in st.session_state:
        st.session_state["watch_history"] = []

    if "last_built_params" not in st.session_state or build_submit:
        st.session_state["last_built_params"] = {"top_n_movies": top_n_movies, "top_n_users": top_n_users}

    last = st.session_state.get("last_built_params", {"top_n_movies": 100, "top_n_users": 500})
    with st.spinner("Building model... please wait"):
        sim_matrix, genres_map, movie_titles, title_to_index = build_recommender(
            top_n_movies=last["top_n_movies"],
            top_n_users=last["top_n_users"],
        )

    genre_options = collect_genre_options(genres_map)
    tmdb_key = _tmdb_api_key()

    with st.sidebar:
        st.divider()
        st.subheader("Wellbeing / content filter")
        st.caption(
            "Movies matching **any** selected tag are hidden from **trending** and **recommendations** "
            "(MovieLens genre labels)."
        )
        if genre_options:
            excluded_genre_pick = st.multiselect(
                "Exclude genres",
                options=genre_options,
                default=[],
            )
        else:
            st.caption("No genre tags in the loaded data — filter unavailable.")
            excluded_genre_pick = []
        excluded_frozen = frozenset(excluded_genre_pick)

        st.divider()
        st.subheader("Mood Mode 🎭")
        st.caption("Boosts movies that match your current mood.")
        mood_pick = st.selectbox(
            "How are you feeling?",
            options=["No preference"] + list(MOOD_GENRE_MAP.keys()),
            index=0,
        )
        mood_genres = MOOD_GENRE_MAP.get(mood_pick, set())

        st.divider()
        st.subheader("TMDB posters")
        if tmdb_key:
            st.success("API key found (posters enabled).")
        else:
            st.warning("No key: set `TMDB_API_KEY` in secrets or environment.")
            st.caption("Streamlit Cloud: App settings → Secrets → `TMDB_API_KEY = \"...\"`")

    tab1, tab2 = st.tabs(["🎬 Recommendations", "📊 Dashboard & Analysis"])

    with tab2:
        st.subheader("📊 Dataset Dashboard")
        st.caption("Analysis of the MovieLens training subset.")

        import plotly.express as px

        df_dash, _, _, _ = load_and_prepare_data(
            top_n_movies=last["top_n_movies"],
            top_n_users=last["top_n_users"],
        )

        st.markdown("#### ⭐ Rating Distribution")
        rating_counts = df_dash["rating"].value_counts().sort_index().reset_index()
        rating_counts.columns = ["Rating", "Count"]
        fig1 = px.bar(
            rating_counts,
            x="Rating",
            y="Count",
            color="Count",
            color_continuous_scale="Blues",
            title="How users rated movies",
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("#### 🎬 Top 10 Most Rated Movies")
        top_movies_df = df_dash["title"].value_counts().head(10).reset_index()
        top_movies_df.columns = ["Movie", "Number of Ratings"]
        fig2 = px.bar(
            top_movies_df,
            x="Number of Ratings",
            y="Movie",
            orientation="h",
            color="Number of Ratings",
            color_continuous_scale="Teal",
            title="Most rated movies in training subset",
        )
        fig2.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### 🎭 Genre Distribution")
        if genres_map:
            all_genres: List[str] = []
            for g in genres_map.values():
                all_genres.extend(genres_string_to_set(g))
            genre_series = pd.Series(all_genres).value_counts().reset_index()
            genre_series.columns = ["Genre", "Count"]
            fig3 = px.pie(
                genre_series,
                names="Genre",
                values="Count",
                title="Genre breakdown across dataset",
                hole=0.3,
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No genre data available.")

        st.markdown("#### 🧠 Sentiment Distribution Across Genres")
        if genres_map:
            sentiment_labels = [genre_sentiment_label(g) for g in genres_map.values()]
            sentiment_df = pd.Series(sentiment_labels).value_counts().reset_index()
            sentiment_df.columns = ["Sentiment", "Count"]
            fig4 = px.bar(
                sentiment_df,
                x="Sentiment",
                y="Count",
                color="Sentiment",
                color_discrete_map={
                    "Positive 😊": "#2ecc71",
                    "Neutral 😐": "#95a5a6",
                    "Negative 😟": "#e74c3c",
                },
                title="Emotional tone of movies in dataset",
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No genre data for sentiment analysis.")

    with tab1:
        import plotly.express as px

        df_ratings, _, _, _ = load_and_prepare_data(
            top_n_movies=last["top_n_movies"],
            top_n_users=last["top_n_users"],
        )
        top_counts = df_ratings["title"].value_counts().head(10)
        trending = pd.DataFrame(
            {
                "Movie": top_counts.index,
                "Ratings in subset": top_counts.values.astype(int),
            }
        )
        trending["Year"] = trending["Movie"].map(extract_year_from_title)
        trending["Genres"] = trending["Movie"].map(lambda t: genres_map.get(t, "—"))
        if excluded_frozen:
            keep = trending["Genres"].map(
                lambda gs: not movie_matches_excluded_genres(str(gs), excluded_frozen)
            )
            trending = trending[keep]
            if trending.empty:
                trending = pd.DataFrame(columns=["Movie", "Ratings in subset", "Year", "Genres"])

        st.subheader("Trending in this dataset")
        st.caption("Most-rated titles in the training subset (respects genre exclusions).")
        if trending.empty:
            st.info("No trending rows left with the current genre exclusions.")
        else:
            st.table(trending)

        st.divider()

        all_sorted = sorted(movie_titles.tolist())
        search_q = st.text_input(
            "Search movies",
            value="",
            placeholder="Type part of a title (e.g. Toy Story)…",
        )
        q = search_q.strip().lower()
        filtered_titles = [t for t in all_sorted if q in t.lower()] if q else all_sorted
        if not filtered_titles:
            st.warning("No matches — reset search to see the full list.")
            filtered_titles = all_sorted

        selected_movie = st.selectbox(
            "Pick a movie",
            options=filtered_titles,
            index=None,
            placeholder="Select a movie...",
        )

        if selected_movie:
            g = genres_map.get(selected_movie, "—") or "—"
            y = extract_year_from_title(selected_movie)
            pc, mc = st.columns([1, 2])
            with pc:
                show_poster_for_title(tmdb_key, selected_movie)
            with mc:
                st.write(f"**Year:** {y}")
                st.write(f"**Genres:** {g}")
        else:
            st.info("Search and select a movie, then press **Recommend**.")

        if selected_movie:
            st.divider()
            st.subheader("🎭 Audience Sentiment Analysis")
            st.caption("Sentiment analysis of audience reviews using TextBlob NLP.")

            from textblob import TextBlob

            reviews = SAMPLE_REVIEWS.get(selected_movie, SAMPLE_REVIEWS["default"])

            is_default_reviews = selected_movie not in SAMPLE_REVIEWS
            if is_default_reviews:
                st.warning(
                    "⚠️ No specific reviews are available for this title. "
                    "The sentiment analysis below uses **generic placeholder reviews** "
                    "and does **not** reflect actual audience opinions for this movie."
                )

            scored = []
            for r in reviews:
                pol = TextBlob(r).sentiment.polarity
                if pol > 0.05:
                    label = "Positive 😊"
                elif pol < -0.05:
                    label = "Negative 😟"
                else:
                    label = "Neutral 😐"
                scored.append({"Review": r, "Sentiment": label, "Polarity": round(pol, 3)})

            scored_df = pd.DataFrame(scored)

            st.markdown("**📋 Sample Reviews**")
            st.table(scored_df)

            sentiment_counts = scored_df["Sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]
            fig_sent = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Count",
                color="Sentiment",
                color_discrete_map={
                    "Positive 😊": "#2ecc71",
                    "Neutral 😐": "#95a5a6",
                    "Negative 😟": "#e74c3c",
                },
                title="Sentiment Distribution",
                hole=0.3,
            )
            st.plotly_chart(fig_sent, use_container_width=True)

            avg_pol = scored_df["Polarity"].mean()
            overall = "Positive 😊" if avg_pol > 0.05 else ("Negative 😟" if avg_pol < -0.05 else "Neutral 😐")
            col1, col2 = st.columns(2)
            col1.metric("Overall Audience Mood", overall)
            col2.metric("Avg Polarity Score", f"{avg_pol:.2f} / 1.0")

            st.markdown("**✍️ Analyse Your Own Review**")
            user_review = st.text_area(
                "Type a review for this movie:",
                placeholder="e.g. This movie was incredible, loved the acting and storyline!",
                height=100,
            )
            if user_review.strip():
                user_pol = TextBlob(user_review).sentiment.polarity
                if user_pol > 0.05:
                    user_label = "Positive 😊"
                elif user_pol < -0.05:
                    user_label = "Negative 😟"
                else:
                    user_label = "Neutral 😐"
                st.success(f"**Your review sentiment:** {user_label} (Polarity: {user_pol:.3f})")

        st.divider()
        st.subheader("Top Recommendations")

        if st.button("Recommend", type="primary"):
            if selected_movie:
                with st.spinner("Generating recommendations..."):
                    recs = recommend_from_similarity(
                        sim_matrix=sim_matrix,
                        movie_titles=movie_titles,
                        movie_name=selected_movie,
                        genres_map=genres_map,
                        title_to_index=title_to_index,
                        top_k=top_k,
                        excluded_genres=excluded_frozen,
                    )

                history: List[str] = st.session_state["watch_history"]
                if selected_movie not in history:
                    history.insert(0, selected_movie)
                    st.session_state["watch_history"] = history[:5]

                st.success("Recommendations generated successfully!")

                if mood_pick != "No preference":
                    st.info(f"🎭 Mood Mode active: **{mood_pick}** — matching movies ranked first.")

                if mood_genres and not recs.empty:
                    recs["mood_match"] = recs["genres"].apply(
                        lambda g: bool(genres_string_to_set(g) & mood_genres)
                    )
                    recs = pd.concat([
                        recs[recs["mood_match"]],
                        recs[~recs["mood_match"]]
                    ]).drop(columns=["mood_match"]).reset_index(drop=True)

                st.markdown(
                    f"**Why these movies?** They are **most similar** to **{selected_movie}** in the training data: "
                    "users who rated your pick tended to rate these titles in a similar way. "
                    "**Similarity** is cosine similarity between movie rating vectors (closer to **1** means more alike)."
                )
                if excluded_genre_pick:
                    st.caption(f"Genre filter active — excluded: {', '.join(sorted(excluded_genre_pick))}")
                if recs.empty:
                    st.warning(
                        "No rows after filtering. Loosen **Exclude genres** or pick another anchor movie."
                    )
                else:
                    display = pd.DataFrame(
                        {
                            "Movie": recs["movie"],
                            "Similarity": recs["score"].round(4),
                            "Year": recs["movie"].map(extract_year_from_title),
                            "Genres": recs["genres"].replace("", "—"),
                            "Sentiment": recs["genres"].apply(genre_sentiment_label),
                        }
                    )
                    st.table(display)
                    n_post = min(5, len(recs))
                    if tmdb_key and n_post:
                        st.subheader("Posters — top suggestions")
                        cols = st.columns(n_post)
                        for j in range(n_post):
                            title_j = str(recs["movie"].iloc[j])
                            with cols[j]:
                                show_poster_for_title(tmdb_key, title_j, title_j[:42])
            else:
                st.warning("Please select a movie first.")

        history = st.session_state.get("watch_history", [])
        if history:
            st.divider()
            st.subheader("🕓 Your Recent Picks")
            st.caption("Movies you explored this session.")
            st.write(" · ".join(history))

            st.subheader("🎯 Based on Your Recent Picks")
            st.caption("Blended recommendations from your last selections.")

            all_recs: List[pd.DataFrame] = []
            for h_title in history:
                if h_title in title_to_index:
                    h_recs = recommend_from_similarity(
                        sim_matrix=sim_matrix,
                        movie_titles=movie_titles,
                        movie_name=h_title,
                        genres_map=genres_map,
                        title_to_index=title_to_index,
                        top_k=20,
                        excluded_genres=excluded_frozen,
                    )
                    all_recs.append(h_recs)

            if all_recs:
                combined = pd.concat(all_recs)
                combined = combined[~combined["movie"].isin(history)]
                combined = (
                    combined.groupby("movie", as_index=False)
                    .agg({"score": "mean", "genres": "first"})
                    .sort_values("score", ascending=False)
                    .head(10)
                    .reset_index(drop=True)
                )
                display_history = pd.DataFrame({
                    "Movie": combined["movie"],
                    "Avg Similarity": combined["score"].round(4),
                    "Year": combined["movie"].map(extract_year_from_title),
                    "Genres": combined["genres"].replace("", "—"),
                    "Sentiment": combined["genres"].apply(genre_sentiment_label),
                })
                st.table(display_history)
            else:
                st.info("No blended recommendations yet — explore more movies.")

        st.markdown("---")
        st.write("RecoMind | Built by Smit Patel 🚀")


if __name__ == "__main__":
    main()
