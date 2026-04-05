import os
import re
import zipfile
import urllib.request
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "Data", "MovieLens")

MOVIES_CSV_LOCAL = os.path.join(DATA_DIR, "movies.csv")
RATINGS_CSV_LOCAL = os.path.join(DATA_DIR, "ratings.csv")

FILTERED_DATA_CSV_LOCAL = os.path.join(DATA_DIR, "filtered_data.csv")
FILTERED_MOVIES_DATA_CSV_LOCAL = os.path.join(DATA_DIR, "filtered_movies_data.csv")

ML_LATEST_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ML_LATEST_SMALL_DIR = os.path.join(DATA_DIR, "ml-latest-small")


def _ensure_data_downloaded() -> Tuple[str, str]:
    """
    Ensures we have a MovieLens movies.csv and ratings.csv available locally.

    If your large local CSVs exist, we use them.
    Otherwise, we download the smaller `ml-latest-small` dataset for GitHub deploys.
    """
    # Use user's large local dataset if present
    if os.path.exists(MOVIES_CSV_LOCAL) and os.path.exists(RATINGS_CSV_LOCAL):
        return MOVIES_CSV_LOCAL, RATINGS_CSV_LOCAL

    os.makedirs(DATA_DIR, exist_ok=True)

    # If the smaller dataset already exists, use it
    extracted_movies = os.path.join(ML_LATEST_SMALL_DIR, "movies.csv")
    extracted_ratings = os.path.join(ML_LATEST_SMALL_DIR, "ratings.csv")
    if os.path.exists(extracted_movies) and os.path.exists(extracted_ratings):
        return extracted_movies, extracted_ratings

    # Download and extract
    zip_path = os.path.join(DATA_DIR, "ml-latest-small.zip")
    with st.spinner("Downloading MovieLens (ml-latest-small) ..."):
        urllib.request.urlretrieve(ML_LATEST_SMALL_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    # After extraction, paths should exist
    if not (os.path.exists(extracted_movies) and os.path.exists(extracted_ratings)):
        raise FileNotFoundError("MovieLens download completed, but CSVs were not found.")

    return extracted_movies, extracted_ratings


@st.cache_data(show_spinner=False)
def load_and_prepare_data(
    top_n_movies: int,
    top_n_users: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    Builds the user-movie rating matrix using the same approach as the notebook:
    - merge ratings + movies
    - filter to top N movies by rating frequency
    - filter to top N active users by rating frequency
    - create a user x movie matrix and fill missing with 0
    """
    # Fast path: if you already created a filtered CSV (recommended),
    # we avoid re-merging + re-filtering at runtime.
    filtered_source = None
    # Prefer the small, deploy-friendly file first.
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

        # If genres are missing from the file, we'll still build the matrix.
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

        # `movies` is only used for genres_map; keep placeholder
        movies = df[["title"]].drop_duplicates().rename(columns={"title": "title"})
        return df, movies, user_movie_matrix, genres_map

    # Default path: use (downloaded or local) MovieLens CSVs and filter at runtime.
    movies_path, ratings_path = _ensure_data_downloaded()

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    df = pd.merge(ratings, movies, on="movieId")
    df.dropna(inplace=True)

    # Strong filtering to keep the matrix small and stable
    top_movies = df["title"].value_counts().head(top_n_movies).index
    df = df[df["title"].isin(top_movies)]

    active_users = df["userId"].value_counts().head(top_n_users).index
    df = df[df["userId"].isin(active_users)]

    # Genres map for display
    genres_map = (
        movies[["title", "genres"]]
        .drop_duplicates(subset=["title"])
        .set_index("title")["genres"]
        .to_dict()
    )

    # user x title matrix with missing ratings set to 0
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
    """
    Returns:
    - sim_matrix: cosine similarity between movies (movies x movies)
    - genres_map: title -> genres string
    - movie_titles: index of movies corresponding to sim_matrix rows/cols
    """
    _, _, user_movie_matrix, genres_map = load_and_prepare_data(
        top_n_movies=top_n_movies, top_n_users=top_n_users
    )

    # Build cosine similarity matrix once (fast recommendations afterwards).
    # user_movie_matrix: shape (users, movies). We want similarity between movie columns.
    X = user_movie_matrix.to_numpy(dtype=np.float32)  # (U, M)
    col_norms = np.linalg.norm(X, axis=0)  # (M,)
    col_norms[col_norms == 0] = 1e-8
    Xn = X / col_norms  # (U, M)
    sim_matrix = Xn.T @ Xn  # (M, M)

    movie_titles = user_movie_matrix.columns
    title_to_index = {title: i for i, title in enumerate(movie_titles)}
    return sim_matrix, genres_map, movie_titles, title_to_index


def extract_year_from_title(title: str) -> str:
    m = re.search(r"\((\d{4})\)\s*$", str(title))
    return m.group(1) if m else "—"


def recommend_from_similarity(
    sim_matrix: np.ndarray,
    movie_titles: pd.Index,
    genres_map: Dict[str, str],
    title_to_index: Dict[str, int],
    movie_name: str,
    top_k: int,
) -> pd.DataFrame:
    if movie_name not in movie_titles:
        return pd.DataFrame([{"movie": movie_name, "score": 0.0, "genres": genres_map.get(movie_name, "")}])

    movie_index = title_to_index[movie_name]
    scores = sim_matrix[movie_index]  # (M,)

    # Fast top-k selection: sort by score desc and skip the movie itself.
    order = np.argsort(-scores)
    top_indices = [i for i in order.tolist() if i != movie_index][:top_k]

    rec_titles = movie_titles[top_indices]
    rec_scores = scores[top_indices]
    rec_df = pd.DataFrame({"movie": rec_titles, "score": rec_scores})
    rec_df["genres"] = rec_df["movie"].map(lambda t: genres_map.get(t, ""))
    return rec_df


def main() -> None:
    st.set_page_config(page_title="RecoMind", page_icon="🎬", layout="centered")
    st.title("RecoMind - Movie Recommendations")
    st.caption("Collaborative filtering using cosine similarity over user ratings.")
    st.info(
        "Pick a movie you like, then tap **Recommend**. "
        "We suggest titles that received **similar rating patterns** from users in the dataset."
    )

    with st.sidebar:
        st.header("Recommendation Settings")
        if os.path.exists(FILTERED_DATA_CSV_LOCAL) or os.path.exists(FILTERED_MOVIES_DATA_CSV_LOCAL):
            st.markdown(
                "**Speed mode**: Found a filtered CSV in `Data/MovieLens` (filters are loaded directly)."
            )
        else:
            st.markdown(
                "**Speed mode**: If you create `Data/MovieLens/filtered_data.csv` (or `filtered_movies_data.csv`), model building becomes much faster."
            )

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

    # Cache rebuild is keyed by these args; using a button prevents constant retraining on every rerun.
    if "last_built_params" not in st.session_state or build_submit:
        st.session_state["last_built_params"] = {"top_n_movies": top_n_movies, "top_n_users": top_n_users}

    last = st.session_state.get("last_built_params", {"top_n_movies": 100, "top_n_users": 500})
    with st.spinner("Building model... please wait"):
        sim_matrix, genres_map, movie_titles, title_to_index = build_recommender(
            top_n_movies=last["top_n_movies"],
            top_n_users=last["top_n_users"],
        )

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

    st.subheader("Trending in this dataset")
    st.caption("Most-rated titles among the movies/users used to train the model (popularity in your filtered data).")
    st.dataframe(trending, use_container_width=True, hide_index=True)

    st.divider()

    selected_movie = st.selectbox(
        "Pick a movie",
        options=sorted(movie_titles.tolist()),
        index=None,
        placeholder="Select a movie...",
    )

    if selected_movie:
        g = genres_map.get(selected_movie, "—") or "—"
        y = extract_year_from_title(selected_movie)
        st.write(f"**Year:** {y}")
        st.write(f"**Genres:** {g}")
    else:
        st.info("Select a movie from the list, then press **Recommend** for personalized suggestions.")

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
                )

            st.success("Recommendations generated successfully!")
            st.markdown(
                f"**Why these movies?** They are **most similar** to **{selected_movie}** in the training data: "
                "users who rated your pick tended to rate these titles in a similar way. "
                "**Similarity** is cosine similarity between movie rating vectors (closer to **1** means more alike)."
            )
            display = pd.DataFrame(
                {
                    "Movie": recs["movie"],
                    "Similarity": recs["score"].round(4),
                    "Year": recs["movie"].map(extract_year_from_title),
                    "Genres": recs["genres"].replace("", "—"),
                }
            )
            st.dataframe(display, use_container_width=True, hide_index=True)
        else:
            st.warning("Please select a movie first.")

    st.markdown("---")
    st.write("RecoMind | Built by Smit Patel 🚀")


if __name__ == "__main__":
    main()

