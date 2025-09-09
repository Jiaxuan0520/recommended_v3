import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Minimal, pure item-based KNN collaborative filtering without extra calculations


@st.cache_data
def load_user_ratings():
    # First try session state if available
    try:
        if 'user_ratings_df' in st.session_state:
            df = st.session_state['user_ratings_df']
            if df is not None and not df.empty:
                return df
    except Exception:
        pass
    # Fallback to local CSV
    try:
        return pd.read_csv('user_movie_rating.csv')
    except Exception:
        return None


def _build_user_item_matrix(ratings_df: pd.DataFrame, movie_ids: np.ndarray):
    if ratings_df is None or ratings_df.empty:
        return None
    ratings = ratings_df[ratings_df['Movie_ID'].isin(movie_ids)].copy()
    if ratings.empty:
        return None
    user_item = ratings.pivot_table(index='User_ID', columns='Movie_ID', values='Rating')
    return user_item


def _fit_item_knn(user_item: pd.DataFrame):
    if user_item is None or user_item.empty:
        return None
    item_vectors = user_item.fillna(0.0).T
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(item_vectors)
    return model, item_vectors


def _nearest_items(model, item_vectors, target_movie_id: int, k: int = 10):
    if model is None or item_vectors is None or target_movie_id not in item_vectors.index:
        return {}
    idx = item_vectors.index.get_loc(target_movie_id)
    distances, indices = model.kneighbors(item_vectors.iloc[[idx]], n_neighbors=min(k + 1, len(item_vectors)))
    neighbors = {}
    for d, i in zip(distances[0], indices[0]):
        nb_movie = int(item_vectors.index[i])
        if nb_movie == target_movie_id:
            continue
        neighbors[nb_movie] = 1.0 - float(d)
    return neighbors


@st.cache_data
def collaborative_knn(merged_df: pd.DataFrame, target_movie: str, top_n: int = 8, k_neighbors: int = 20):
    """Item-based CF using user ratings for similarity; rank neighbors by average user rating.

    - Similarity: cosine on item vectors from user-item matrix
    - Ranking: by average user rating (from user_movie_rating.csv), then number of ratings, then similarity
    - Uses only imdb_top_1000.csv (metadata) and user_movie_rating.csv (interactions)
    """
    if target_movie is None or not isinstance(target_movie, str) or target_movie.strip() == '':
        return None

    if 'Movie_ID' not in merged_df.columns or 'Series_Title' not in merged_df.columns:
        return None

    # Map titles to Movie_ID
    title_to_id = dict(merged_df[['Series_Title', 'Movie_ID']].values)
    if target_movie in title_to_id:
        target_movie_id = int(title_to_id[target_movie])
    else:
        match_series = merged_df[merged_df['Series_Title'].str.lower() == target_movie.lower()]
        if match_series.empty:
            return None
        target_movie_id = int(match_series.iloc[0]['Movie_ID'])

    ratings_df = load_user_ratings()
    if ratings_df is None or ratings_df.empty:
        return None

    user_item = _build_user_item_matrix(ratings_df, merged_df['Movie_ID'].values)
    model, item_vectors = _fit_item_knn(user_item)
    neighbors = _nearest_items(model, item_vectors, target_movie_id, k=k_neighbors)
    if not neighbors:
        return None

    # Compute average user rating and count per movie
    agg = ratings_df.groupby('Movie_ID')['Rating'].agg(['mean', 'count']).reset_index()
    agg = agg.rename(columns={'mean': 'Avg_User_Rating', 'count': 'Num_Ratings'})

    # Build candidate table
    cand = pd.DataFrame({'Movie_ID': list(neighbors.keys())})
    cand['Similarity'] = cand['Movie_ID'].map(neighbors)
    cand = cand.merge(agg, on='Movie_ID', how='left')
    # If some movies have no ratings in file (unlikely), drop or fill with global mean
    if cand['Avg_User_Rating'].isna().any():
        global_mean = float(ratings_df['Rating'].mean())
        cand['Avg_User_Rating'] = cand['Avg_User_Rating'].fillna(global_mean)
        cand['Num_Ratings'] = cand['Num_Ratings'].fillna(0)

    # Rank: average rating desc, then count desc, then similarity desc
    cand = cand.sort_values(['Avg_User_Rating', 'Num_Ratings', 'Similarity'], ascending=[False, False, False])
    cand = cand.head(top_n)

    # Map to titles and attach metadata for display
    result = cand.merge(merged_df[['Movie_ID', 'Series_Title']], on='Movie_ID', how='left')
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else ('Rating' if 'Rating' in merged_df.columns else None)
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else ('Genre' if 'Genre' in merged_df.columns else None)
    cols = ['Movie_ID', 'Series_Title'] + ([genre_col] if genre_col else []) + ([rating_col] if rating_col else [])
    result = result.merge(merged_df[cols].drop_duplicates(['Series_Title','Movie_ID']), on=['Series_Title','Movie_ID'], how='left')

    # Final order preserved from ranking
    # Drop Movie_ID for cleaner display in app
    display_cols = ['Series_Title']
    if genre_col:
        display_cols.append(genre_col)
    if rating_col:
        display_cols.append(rating_col)
    display_cols += ['Avg_User_Rating', 'Num_Ratings', 'Similarity']
    # Ensure columns exist
    result = result[display_cols]
    return result


@st.cache_data
def collaborative_rank_by_avg_user_rating(merged_df: pd.DataFrame, target_movie: str, top_n: int = 8, k_neighbors: int = 20):
    """Safer alias for the collaborative algorithm to avoid any legacy cache path issues."""
    return collaborative_knn(merged_df, target_movie, top_n=top_n, k_neighbors=k_neighbors)


@st.cache_data
def collaborative_filtering_enhanced(merged_df: pd.DataFrame, target_movie: str, top_n: int = 8):
    # Minimal wrapper to keep existing app API; uses KNN for similarity then rank by avg user rating
    return collaborative_rank_by_avg_user_rating(merged_df, target_movie, top_n=top_n)


@st.cache_data
def diagnose_data_linking(merged_df: pd.DataFrame):
    issues = {}
    issues['has_movie_id'] = 'Movie_ID' in merged_df.columns
    issues['unique_titles'] = merged_df['Series_Title'].nunique()
    issues['rows'] = len(merged_df)
    try:
        ratings = load_user_ratings()
        issues['ratings_loaded'] = ratings is not None and not ratings.empty
        if issues['ratings_loaded'] and issues['has_movie_id']:
            covered = ratings['Movie_ID'].isin(merged_df['Movie_ID']).mean()
            issues['ratings_coverage_ratio'] = float(covered)
    except Exception:
        issues['ratings_loaded'] = False
    return issues