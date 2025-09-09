import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import streamlit as st

# Minimal, pure item-based KNN collaborative filtering without extra calculations


@st.cache_data
def load_user_ratings():
    # First try session state if available
    try:
        if 'user_ratings_df' in st.session_state:
            df = st.session_state['user_ratings_df']
            if df is not None and not df.empty:
                # Normalize column names: prefer Movie_ID; rename Series_ID -> Movie_ID if needed
                if 'Movie_ID' not in df.columns and 'Series_ID' in df.columns:
                    df = df.rename(columns={'Series_ID': 'Movie_ID'})
                # Normalize dtypes to avoid ID mismatches (string vs int)
                for col in ['User_ID', 'Movie_ID', 'Rating']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=['User_ID', 'Movie_ID', 'Rating'])
                df['User_ID'] = df['User_ID'].astype(int)
                df['Movie_ID'] = df['Movie_ID'].astype(int)
                df['Rating'] = df['Rating'].astype(float)
                return df
    except Exception:
        pass
    # Fallback to local CSV
    try:
        df = pd.read_csv('user_movie_rating.csv')
        if 'Movie_ID' not in df.columns and 'Series_ID' in df.columns:
            df = df.rename(columns={'Series_ID': 'Movie_ID'})
        for col in ['User_ID', 'Movie_ID', 'Rating']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['User_ID', 'Movie_ID', 'Rating'])
        df['User_ID'] = df['User_ID'].astype(int)
        df['Movie_ID'] = df['Movie_ID'].astype(int)
        df['Rating'] = df['Rating'].astype(float)
        return df
    except Exception:
        return None


def _build_user_item_matrix(ratings_df: pd.DataFrame, movie_ids: np.ndarray):
    if ratings_df is None or ratings_df.empty:
        return None
    # Ensure comparable dtypes to avoid false negatives in isin
    ratings_df = ratings_df.copy()
    ratings_df['Movie_ID'] = pd.to_numeric(ratings_df['Movie_ID'], errors='coerce').astype('Int64')
    movie_ids = pd.Series(movie_ids).astype('Int64')
    ratings = ratings_df[ratings_df['Movie_ID'].isin(set(movie_ids.dropna().tolist()))].copy()
    if ratings.empty:
        return None
    user_item = ratings.pivot_table(index='User_ID', columns='Movie_ID', values='Rating')
    return user_item


def _fit_item_knn(user_item: pd.DataFrame):
    if user_item is None or user_item.empty:
        return None
    item_vectors = user_item.fillna(0.0).T
    # Use Euclidean distance for pure KNN (no cosine similarity)
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
        # Convert Euclidean distance to a bounded similarity in (0,1]
        sim = 1.0 / (1.0 + float(d))
        neighbors[nb_movie] = sim
    return neighbors


def _genre_column_name(df: pd.DataFrame):
    if 'Genre_y' in df.columns:
        return 'Genre_y'
    if 'Genre' in df.columns:
        return 'Genre'
    return None


def _split_genres(genre_text):
    if pd.isna(genre_text):
        return set()
    return set([g.strip().lower() for g in str(genre_text).split(',') if str(g).strip()])


def _select_proxy_movie_id(merged_df: pd.DataFrame, user_item: pd.DataFrame, target_movie_id: int, ratings_df: pd.DataFrame):
    """If target item has no ratings, choose a close rated item to act as proxy.
    Preference: shared genres with target, then higher rating count in user data.
    """
    if user_item is None or user_item.empty:
        return None
    available_ids = set(user_item.columns)
    if target_movie_id in available_ids:
        return target_movie_id
    if not available_ids:
        return None
    genre_col = _genre_column_name(merged_df)
    target_row = merged_df[merged_df['Movie_ID'] == target_movie_id]
    target_genres = _split_genres(target_row.iloc[0][genre_col]) if (genre_col and not target_row.empty) else set()
    candidates = merged_df[merged_df['Movie_ID'].isin(available_ids)].copy()
    if candidates.empty:
        return None
    # Precompute rating counts
    counts = ratings_df['Movie_ID'].value_counts() if (ratings_df is not None and not ratings_df.empty and 'Movie_ID' in ratings_df.columns) else None
    def compute_score(row):
        genres = _split_genres(row.get(genre_col, '')) if genre_col else set()
        overlap = len(genres & target_genres)
        cnt = int(counts.get(row['Movie_ID'], 0)) if counts is not None else 0
        return overlap, cnt
    scores = candidates.apply(lambda r: compute_score(r), axis=1)
    candidates['__overlap__'] = [s[0] for s in scores]
    candidates['__cnt__'] = [s[1] for s in scores]
    candidates = candidates.sort_values(['__overlap__', '__cnt__'], ascending=[False, False])
    return int(candidates.iloc[0]['Movie_ID']) if not candidates.empty else None


def _fallback_cooccurrence_neighbors(ratings_df: pd.DataFrame, seed_movie_id: int, allowed_movie_ids: set, max_candidates: int = 50):
    """Fallback based on co-occurrence: items co-rated by users who rated the seed item.
    Returns dict movie_id -> similarity in [0,1].
    """
    if ratings_df is None or ratings_df.empty or seed_movie_id is None:
        return {}
    if 'Movie_ID' not in ratings_df.columns:
        return {}
    df = ratings_df[ratings_df['Movie_ID'].isin(allowed_movie_ids)].copy()
    seed_users = df[df['Movie_ID'] == seed_movie_id]['User_ID'].unique()
    if len(seed_users) == 0:
        return {}
    co = df[(df['User_ID'].isin(seed_users)) & (df['Movie_ID'] != seed_movie_id)]
    if co.empty:
        return {}
    counts = co.groupby('Movie_ID').size().sort_values(ascending=False)
    if counts.empty:
        return {}
    max_cnt = float(counts.iloc[0])
    neighbors = {}
    for mid, cnt in counts.head(max_candidates).items():
        neighbors[int(mid)] = float(cnt) / max_cnt if max_cnt > 0 else 0.0
    return neighbors


def _fallback_popularity_neighbors(ratings_df: pd.DataFrame, allowed_movie_ids: set, seed_movie_id: int, max_candidates: int = 50, merged_df: pd.DataFrame = None):
    """Fallback based on popularity in ratings, preferring same-genre items when possible."""
    if ratings_df is None or ratings_df.empty:
        return {}
    if 'Movie_ID' not in ratings_df.columns:
        return {}
    df = ratings_df[ratings_df['Movie_ID'].isin(allowed_movie_ids)].copy()
    if df.empty:
        return {}
    prefer_ids = allowed_movie_ids
    # Prefer same-genre items if metadata available
    if merged_df is not None and seed_movie_id is not None and 'Movie_ID' in merged_df.columns:
        genre_col = _genre_column_name(merged_df)
        if genre_col:
            target_row = merged_df[merged_df['Movie_ID'] == seed_movie_id]
            if not target_row.empty:
                target_genres = _split_genres(target_row.iloc[0][genre_col])
                if target_genres:
                    same_genre_ids = set()
                    for _, row in merged_df.iterrows():
                        if row['Movie_ID'] in allowed_movie_ids:
                            g = _split_genres(row.get(genre_col, ''))
                            if len(g & target_genres) > 0:
                                same_genre_ids.add(int(row['Movie_ID']))
                    if len(same_genre_ids) > 0:
                        prefer_ids = same_genre_ids
    df = df[df['Movie_ID'].isin(prefer_ids)]
    if df.empty:
        return {}
    counts = df['Movie_ID'].value_counts()
    max_cnt = float(counts.iloc[0]) if not counts.empty else 0.0
    neighbors = {}
    for mid, cnt in counts.head(max_candidates).items():
        if int(mid) == int(seed_movie_id):
            continue
        neighbors[int(mid)] = float(cnt) / max_cnt if max_cnt > 0 else 0.0
    return neighbors


@st.cache_data
def collaborative_knn(merged_df: pd.DataFrame, target_movie: str, top_n: int = 8, k_neighbors: int = 20):
    if target_movie is None or not isinstance(target_movie, str) or target_movie.strip() == '':
        return None

    if 'Movie_ID' not in merged_df.columns or 'Series_Title' not in merged_df.columns:
        return None

    # Normalize dtypes to ensure ID matches with ratings
    merged_df = merged_df.copy()
    merged_df['Movie_ID'] = pd.to_numeric(merged_df['Movie_ID'], errors='coerce').astype('Int64')
    merged_df = merged_df.dropna(subset=['Movie_ID'])
    merged_df['Movie_ID'] = merged_df['Movie_ID'].astype(int)
    merged_df['Series_Title'] = merged_df['Series_Title'].astype(str)

    # Map titles to Movie_ID
    title_to_id = dict(merged_df[['Series_Title', 'Movie_ID']].values)
    if target_movie not in title_to_id:
        # try case-insensitive
        match_series = merged_df[merged_df['Series_Title'].str.lower() == target_movie.lower()]
        if match_series.empty:
            return None
        target_movie_id = int(match_series.iloc[0]['Movie_ID'])
    else:
        target_movie_id = int(title_to_id[target_movie])

    ratings_df = load_user_ratings()
    user_item = _build_user_item_matrix(ratings_df, merged_df['Movie_ID'].values)
    model, item_vectors = _fit_item_knn(user_item)

    # If target has no ratings, try a proxy rated movie with similar genre to seed neighbors
    proxy_movie_id = target_movie_id
    if (item_vectors is None) or (target_movie_id not in (item_vectors.index if item_vectors is not None else [])):
        proxy_movie_id = _select_proxy_movie_id(merged_df, user_item, target_movie_id, ratings_df)
    neighbors = _nearest_items(model, item_vectors, proxy_movie_id, k=k_neighbors)
    if not neighbors:
        # Fallback 1: co-occurrence among users who rated the proxy
        allowed_ids = set(pd.Series(merged_df['Movie_ID']).dropna().astype(int).tolist())
        neighbors = _fallback_cooccurrence_neighbors(ratings_df, proxy_movie_id, allowed_ids, top_n * 3)
    if not neighbors:
        # Fallback 2: overall popularity from ratings (prefer same-genre)
        allowed_ids = set(pd.Series(merged_df['Movie_ID']).dropna().astype(int).tolist())
        neighbors = _fallback_popularity_neighbors(ratings_df, allowed_ids, proxy_movie_id, top_n * 3, merged_df)
    if not neighbors:
        return None

    # Rank by similarity only (pure KNN)
    sorted_pairs = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
    # Remove the proxy itself if it equals target
    if proxy_movie_id == target_movie_id and target_movie_id in dict(sorted_pairs):
        sorted_pairs = [(mid, sim) for mid, sim in sorted_pairs if mid != target_movie_id]
    sorted_pairs = sorted_pairs[:top_n]
    sorted_ids = [mid for mid, sim in sorted_pairs]
    sim_by_id = {mid: sim for mid, sim in sorted_pairs}
    result = merged_df[merged_df['Movie_ID'].isin(sorted_ids)][['Series_Title', 'Movie_ID']]
    # Keep original rating/genre columns if present
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else ('Rating' if 'Rating' in merged_df.columns else None)
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else ('Genre' if 'Genre' in merged_df.columns else None)
    cols = ['Series_Title', 'Movie_ID'] + ([genre_col] if genre_col else []) + ([rating_col] if rating_col else [])
    result = result.merge(merged_df[cols].drop_duplicates(['Series_Title','Movie_ID']), on=['Series_Title','Movie_ID'], how='left')

    # Preserve similarity order
    title_by_id = dict(merged_df[['Movie_ID', 'Series_Title']].values)
    order = {title_by_id[mid]: i for i, mid in enumerate(sorted_ids) if mid in title_by_id}
    result = result.copy()
    result['rank_order'] = result['Series_Title'].map(order)
    result['Similarity'] = result['Movie_ID'].map(sim_by_id)
    result = result.sort_values('rank_order').drop(columns=['rank_order'])
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