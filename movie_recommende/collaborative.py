import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import streamlit as st
from difflib import get_close_matches

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
    # Use Euclidean distance as requested
    model = NearestNeighbors(metric='euclidean', algorithm='brute')
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
        # Convert Euclidean distance to a closeness score in (0, 1]
        # Higher is more similar
        neighbors[nb_movie] = 1.0 / (1.0 + float(d))
    return neighbors


def _find_genre_column(df: pd.DataFrame) -> str:
    return 'Genre_y' if 'Genre_y' in df.columns else 'Genre'


def _fuzzy_match_title(input_title: str, titles: pd.Series) -> str:
    if not isinstance(input_title, str) or input_title.strip() == '':
        return None
    unique_titles = titles.dropna().unique().tolist()
    # Exact match
    for t in unique_titles:
        if t == input_title:
            return t
    # Case-insensitive exact
    lowered = {t.lower(): t for t in unique_titles}
    if input_title.lower() in lowered:
        return lowered[input_title.lower()]
    # Close matches
    matches = get_close_matches(input_title, unique_titles, n=1, cutoff=0.6)
    return matches[0] if matches else None


def _genre_based_fallback(merged_df: pd.DataFrame, ratings_df: pd.DataFrame, target_title: str, top_n: int) -> pd.DataFrame:
    if merged_df is None or ratings_df is None or ratings_df.empty:
        return None
    if target_title not in merged_df['Series_Title'].values:
        return None
    target_row = merged_df[merged_df['Series_Title'] == target_title].iloc[0]
    genre_col = _find_genre_column(merged_df)
    target_genre = str(target_row.get(genre_col, '')).strip()
    if not target_genre:
        return None
    # Split target genre into tokens and match any
    target_genres = [g.strip().lower() for g in target_genre.split(',') if g.strip()]
    if not target_genres:
        return None
    def genre_match(g):
        gs = [x.strip().lower() for x in str(g).split(',') if str(g)]
        return any(x in gs for x in target_genres)
    candidates = merged_df[merged_df[genre_col].apply(genre_match)].copy()
    if candidates.empty:
        return None
    # Aggregate user ratings per movie
    user_agg = ratings_df.groupby('Movie_ID')['Rating'].agg(['mean', 'count']).rename(columns={'mean': 'User_Avg_Rating', 'count': 'User_Rating_Count'})
    candidates = candidates.merge(user_agg, left_on='Movie_ID', right_index=True, how='left')
    # Exclude target itself
    candidates = candidates[candidates['Series_Title'] != target_title]
    if candidates.empty:
        return None
    # Build result columns
    keep_cols = ['Series_Title', 'Movie_ID']
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else ('Rating' if 'Rating' in merged_df.columns else None)
    if genre_col:
        keep_cols.append(genre_col)
    if rating_col:
        keep_cols.append(rating_col)
    # Rank by user average rating, then by number of ratings
    candidates['User_Rating_Score'] = (candidates['User_Avg_Rating'] / 10.0).clip(lower=0.0, upper=1.0)
    candidates['Similarity'] = np.nan  # not from KNN
    ranked = candidates.sort_values(by=['User_Avg_Rating', 'User_Rating_Count'], ascending=[False, False]).head(top_n)
    return ranked[keep_cols + ['User_Avg_Rating', 'User_Rating_Count', 'User_Rating_Score', 'Similarity']].drop(columns=['Movie_ID'])


@st.cache_data
def collaborative_knn(merged_df: pd.DataFrame, target_movie: str, top_n: int = 8, k_neighbors: int = 20):
    if target_movie is None or not isinstance(target_movie, str) or target_movie.strip() == '':
        return None

    if 'Movie_ID' not in merged_df.columns or 'Series_Title' not in merged_df.columns:
        return None

    # Map titles to Movie_ID with fuzzy matching
    target_title = _fuzzy_match_title(target_movie, merged_df['Series_Title'])
    if target_title is None:
        return None
    target_movie_id = int(merged_df.loc[merged_df['Series_Title'] == target_title, 'Movie_ID'].iloc[0])

    ratings_df = load_user_ratings()
    user_item = _build_user_item_matrix(ratings_df, merged_df['Movie_ID'].values)
    model, item_vectors = _fit_item_knn(user_item)
    neighbors = _nearest_items(model, item_vectors, target_movie_id, k=k_neighbors)
    # Pre-compute user aggregates (used in both primary and fallback flows)
    user_agg = None
    if ratings_df is not None and not ratings_df.empty and {'Movie_ID', 'Rating'}.issubset(ratings_df.columns):
        user_agg = ratings_df.groupby('Movie_ID')['Rating'].agg(['mean', 'count']).rename(columns={'mean': 'User_Avg_Rating', 'count': 'User_Rating_Count'})
    # If no neighbors (e.g., cold start), use genre-based fallback by user ratings
    if not neighbors:
        return _genre_based_fallback(merged_df, ratings_df, target_title, top_n)

    # Compute user-based aggregate ratings for ranking
    # user_agg already computed above

    # Candidate neighbor IDs
    neighbor_ids = list(neighbors.keys())

    # Build base result with metadata
    base_cols = ['Series_Title', 'Movie_ID']
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else ('Rating' if 'Rating' in merged_df.columns else None)
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else ('Genre' if 'Genre' in merged_df.columns else None)
    if genre_col:
        base_cols.append(genre_col)
    if rating_col:
        base_cols.append(rating_col)

    result = merged_df[merged_df['Movie_ID'].isin(neighbor_ids)][base_cols].drop_duplicates(['Series_Title','Movie_ID'])

    # Attach closeness (similarity-like) score
    result = result.copy()
    result['Similarity'] = result['Movie_ID'].map(neighbors)

    # Attach user rating aggregates
    if user_agg is not None:
        result = result.merge(user_agg, left_on='Movie_ID', right_index=True, how='left')
        # Normalize average user rating to [0,1] if ratings are on 1..10 scale
        result['User_Rating_Score'] = (result['User_Avg_Rating'] / 10.0).clip(lower=0.0, upper=1.0)
    else:
        result['User_Avg_Rating'] = np.nan
        result['User_Rating_Count'] = 0
        result['User_Rating_Score'] = np.nan

    # Rank neighbors primarily by user average rating, then by count, then by closeness
    result = result.sort_values(by=['User_Avg_Rating', 'User_Rating_Count', 'Similarity'], ascending=[False, False, False])

    # Limit to top_n after ranking
    result = result.head(top_n)

    return result.drop(columns=['Movie_ID'])


@st.cache_data
def collaborative_filtering_enhanced(merged_df: pd.DataFrame, target_movie: str, top_n: int = 8):
    # Minimal wrapper to keep existing app API; uses pure KNN ranking
    return collaborative_knn(merged_df, target_movie, top_n=top_n)


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