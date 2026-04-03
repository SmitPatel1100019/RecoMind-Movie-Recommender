# RecoMind - Movie Recommendation System

RecoMind is a collaborative-filtering movie recommender built from MovieLens ratings. This repository exposes the recommender as a web app using **Streamlit**.

## What you get
- A Streamlit UI where you choose a movie title
- Recommendations generated with cosine similarity over a filtered user-movie rating matrix

## Local run
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Start the app:
   - `streamlit run app.py`

## Data (important for GitHub deploy)
- **Recommended (fast + GitHub-friendly):** keep a small merged filtered file at `Data/MovieLens/filtered_movies_data.csv`
  - Required columns: `userId,title,rating` (and optionally `genres`)
  - The app will prefer this file when present (fast startup + fast recommendations).
- If no filtered file exists, the app will:
  - Use `Data/MovieLens/movies.csv` + `Data/MovieLens/ratings.csv` if they exist locally, otherwise
  - Download the smaller `ml-latest-small` dataset automatically at runtime.

## Deploy from GitHub (recommended)
This project is meant to deploy to a Streamlit host that can pull from GitHub (for example, **Streamlit Community Cloud**):
1. Create a GitHub repository and push your code.
2. Connect the repo to your Streamlit hosting service.
3. Set the entry file to `app.py`.

