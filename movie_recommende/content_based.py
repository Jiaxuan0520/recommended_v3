import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from difflib import get_close_matches
import streamlit as st


def safe_convert_to_numeric(value, default=None):
    """Safely convert a value to numeric, handling strings and NaN"""
    if pd.isna(value):
        return default
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # Remove any non-numeric characters except decimal point
        clean_value = re.sub(r'[^\d.-]', '', str(value))
        try:
            return float(clean_value) if clean_value else default
        except (ValueError, TypeError):
            return default
    
    return default


def find_rating_column(df: pd.DataFrame) -> str:
    return 'IMDB_Rating' if 'IMDB_Rating' in df.columns else 'Rating'


def find_genre_column(df: pd.DataFrame) -> str:
    return 'Genre_y' if 'Genre_y' in df.columns else 'Genre'




def find_similar_titles(input_title, titles_list, cutoff=0.6):
    """Enhanced fuzzy matching for movie titles"""
    if not input_title or not titles_list:
        return []
    
    input_lower = input_title.lower().strip()
    
    # Direct match
    exact_matches = [title for title in titles_list if title.lower() == input_lower]
    if exact_matches:
        return exact_matches
    
    # Partial match
    partial_matches = []
    for title in titles_list:
        title_lower = title.lower()
        if input_lower in title_lower:
            partial_matches.append((title, len(input_lower) / len(title_lower)))
        elif title_lower in input_lower:
            partial_matches.append((title, len(title_lower) / len(input_lower)))
            
    if partial_matches:
        # Sort by match ratio
        partial_matches.sort(key=lambda x: x[1], reverse=True)
        return [match[0] for match in partial_matches]

    # Close matches
    return get_close_matches(input_title, titles_list, n=5, cutoff=cutoff)


@st.cache_data
def create_content_features(merged_df):
    """Create enhanced TF-IDF features using multiple movie attributes with optimized weights"""

    genre_col = find_genre_column(merged_df)
    rating_col = find_rating_column(merged_df)
    
    # Find additional columns
    year_col = 'Released_Year' if 'Released_Year' in merged_df.columns else 'Year'
    director_col = 'Director' if 'Director' in merged_df.columns else None
    overview_col = 'Overview' if 'Overview' in merged_df.columns else None
    runtime_col = 'Runtime' if 'Runtime' in merged_df.columns else None
    certificate_col = 'Certificate' if 'Certificate' in merged_df.columns else None

    # Enhanced weights for better feature balance
    WEIGHTS = {
        'title': 2,           
        'genre': 10,         
        'rating': 3,          
        'year': 2,           
        'director': 3,        
        'overview': 5,       
        'runtime': 1,         
        'certificate': 1,     
    }

    def build_row_text(row: pd.Series) -> str:
        features = []
        
        # 1. Title (enhanced)
        title = str(row.get('Series_Title', '')).strip()
        if title:
            features.extend([title] * WEIGHTS['title'])
        
        # 2. Genre (enhanced - split multi-genres)
        genre = str(row.get(genre_col, '')).strip()
        if genre and genre != 'nan':
            # Split genres and create individual tokens
            genres = [g.strip() for g in genre.split(',') if g.strip()]
            for g in genres:
                features.extend([g] * WEIGHTS['genre'])
        
        # 3. Rating (enhanced with more granular buckets)
        rating_val = safe_convert_to_numeric(row.get(rating_col, np.nan), default=np.nan)
        if pd.isna(rating_val):
            rating_val = 7.0
        rating_bucket = int(max(1, min(10, round(rating_val))))
        rating_token = f"rating_{rating_bucket}"
        features.extend([rating_token] * WEIGHTS['rating'])
        
        # 4. Year/Decade features (NEW)
        year = row.get(year_col, None)
        if year is not None:
            try:
                if isinstance(year, str):
                    year_val = int(year.split()[0].strip('()'))
                else:
                    year_val = int(year) if pd.notna(year) else None
                
                if year_val:
                    # Add decade token
                    decade = f"decade_{year_val//10*10}s"
                    features.extend([decade] * WEIGHTS['year'])
                    
                    # Add era token (broader categories)
                    if year_val < 1980:
                        era = "classic_era"
                    elif year_val < 2000:
                        era = "modern_era"
                    else:
                        era = "contemporary_era"
                    features.extend([era] * WEIGHTS['year'])
            except:
                pass
        
        # 5. Director features (NEW)
        if director_col and director_col in row:
            director = str(row.get(director_col, '')).strip()
            if director and director != 'nan':
                # Split multiple directors if any
                directors = [d.strip() for d in director.split(',') if d.strip()]
                for d in directors:
                    features.extend([d] * WEIGHTS['director'])
        
        # 6. Overview/Plot text (NEW)
        if overview_col and overview_col in row:
            overview = str(row.get(overview_col, '')).strip()
            if overview and overview != 'nan':
                # Clean and process overview text
                overview_clean = re.sub(r'[^\w\s]', ' ', overview.lower())
                features.extend([overview_clean] * WEIGHTS['overview'])
        
        # 7. Runtime categories (NEW)
        if runtime_col and runtime_col in row:
            runtime = str(row.get(runtime_col, '')).strip()
            if runtime and runtime != 'nan':
                try:
                    # Extract numeric value from runtime string
                    runtime_match = re.search(r'(\d+)', runtime)
                    if runtime_match:
                        runtime_min = int(runtime_match.group(1))
                        if runtime_min < 90:
                            runtime_cat = "short_film"
                        elif runtime_min < 120:
                            runtime_cat = "standard_film"
                        elif runtime_min < 150:
                            runtime_cat = "long_film"
                        else:
                            runtime_cat = "epic_film"
                        features.extend([runtime_cat] * WEIGHTS['runtime'])
                except:
                    pass
        
        # 8. Certificate/Age rating (NEW)
        if certificate_col and certificate_col in row:
            certificate = str(row.get(certificate_col, '')).strip()
            if certificate and certificate != 'nan':
                features.extend([certificate] * WEIGHTS['certificate'])

        return ' '.join(features)

    merged_df = merged_df.copy()
    merged_df['cb_text'] = merged_df.apply(build_row_text, axis=1)

    # Enhanced TF-IDF parameters for better feature extraction
    tfidf = TfidfVectorizer(
        stop_words='english', 
        ngram_range=(1, 3),  # Extended to 3-grams for better context
        min_df=2,             # Increased to filter rare terms
        max_df=0.95,          # Filter very common terms
        max_features=10000,   # Limit features for performance
        sublinear_tf=True,    # Use sublinear TF scaling
        norm='l2'             # L2 normalization
    )
    return tfidf.fit_transform(merged_df['cb_text'])


@st.cache_data
def content_based_filtering_enhanced(merged_df, target_movie=None, genre=None, top_n=8, 
                                    similarity_threshold=0.1, diversity_factor=0.3):
    """Enhanced Content-Based filtering with similarity thresholds and diversity"""
    # Get column names once at the start
    rating_col = find_rating_column(merged_df)
    genre_col = find_genre_column(merged_df)
    
    if target_movie:
        similar_titles = find_similar_titles(target_movie, merged_df['Series_Title'].tolist())
        if not similar_titles:
            return None
        
        target_title = similar_titles[0]
        
        # Ensure the target movie exists in the dataframe
        if target_title not in merged_df['Series_Title'].values:
            return None
            
        target_idx = merged_df[merged_df['Series_Title'] == target_title].index[0]
        
        content_features = create_content_features(merged_df)
        target_features = content_features[merged_df.index.get_loc(target_idx)].reshape(1, -1)
        similarities = cosine_similarity(target_features, content_features).flatten()
        
        # Apply similarity threshold
        valid_indices = []
        for i, sim in enumerate(similarities):
            if i != merged_df.index.get_loc(target_idx) and sim >= similarity_threshold:
                valid_indices.append((i, sim))
        
        if not valid_indices:
            return None
        
        # Sort by similarity and apply diversity
        valid_indices.sort(key=lambda x: x[1], reverse=True)
        
        # Apply diversity filtering to avoid too similar recommendations
        selected_indices = []
        selected_genres = set()
        
        for idx, sim in valid_indices:
            if len(selected_indices) >= top_n:
                break
                
            movie_genres = str(merged_df.iloc[idx][genre_col]).split(',')
            movie_genres = {g.strip() for g in movie_genres if g.strip()}
            
            # Check diversity: if we have too many movies from same genres, skip
            genre_overlap = len(selected_genres.intersection(movie_genres))
            max_overlap = int(len(selected_genres) * diversity_factor) + 1
            
            if genre_overlap <= max_overlap or len(selected_indices) < 3:
                selected_indices.append(idx)
                selected_genres.update(movie_genres)
        
        if not selected_indices:
            # Fallback: just take top similar movies
            selected_indices = [idx for idx, _ in valid_indices[:top_n]]
        
        result_df = merged_df.iloc[selected_indices]
        return result_df[['Series_Title', genre_col, rating_col]]
    
    elif genre:
        # Enhanced genre filtering with similarity threshold
        genre_corpus = merged_df[genre_col].fillna('').tolist()
        tfidf = TfidfVectorizer(
            stop_words='english', 
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        tfidf_matrix = tfidf.fit_transform(genre_corpus)
        query_vector = tfidf.transform([genre])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Apply similarity threshold
        valid_indices = [(i, sim) for i, sim in enumerate(similarities) if sim >= similarity_threshold]
        
        if not valid_indices:
            return None
        
        # Sort by similarity and apply diversity
        valid_indices.sort(key=lambda x: x[1], reverse=True)
        
        # Apply diversity filtering
        selected_indices = []
        selected_genres = set()
        
        for idx, sim in valid_indices:
            if len(selected_indices) >= top_n:
                break
                
            movie_genres = str(merged_df.iloc[idx][genre_col]).split(',')
            movie_genres = {g.strip() for g in movie_genres if g.strip()}
            
            # Check diversity
            genre_overlap = len(selected_genres.intersection(movie_genres))
            max_overlap = int(len(selected_genres) * diversity_factor) + 1
            
            if genre_overlap <= max_overlap or len(selected_indices) < 3:
                selected_indices.append(idx)
                selected_genres.update(movie_genres)
        
        if not selected_indices:
            selected_indices = [idx for idx, _ in valid_indices[:top_n]]
        
        result_df = merged_df.iloc[selected_indices]
        return result_df[['Series_Title', genre_col, rating_col]]
        
    return None


def content_based_with_signals(merged_df, target_movie=None, genre=None, top_n=8, 
                              content_weight=0.6, popularity_weight=0.2, recency_weight=0.2):
    """Enhanced content-based filtering with popularity and recency signals"""
    # Get column names once at the start
    rating_col = find_rating_column(merged_df)
    genre_col = find_genre_column(merged_df)
    
    def normalize_scores(scores_dict):
        """Normalize scores to 0-1 range"""
        if not scores_dict:
            return {}
        values = list(scores_dict.values())
        if not values:
            return scores_dict
        min_val = min(values)
        max_val = max(values)
        if max_val == min_val:
            return {k: 1.0 for k in scores_dict.keys()}
        return {k: (v - min_val) / (max_val - min_val) for k, v in scores_dict.items()}
    
    # 1. Content-based scores
    content_scores = {}
    if target_movie:
        similar_titles = find_similar_titles(target_movie, merged_df['Series_Title'].tolist())
        if similar_titles:
            target_title = similar_titles[0]
            if target_title in merged_df['Series_Title'].values:
                target_idx = merged_df[merged_df['Series_Title'] == target_title].index[0]
                content_features = create_content_features(merged_df)
                target_features = content_features[merged_df.index.get_loc(target_idx)].reshape(1, -1)
                similarities = cosine_similarity(target_features, content_features).flatten()
                
                for i, sim_score in enumerate(similarities):
                    if i != merged_df.index.get_loc(target_idx):
                        title = merged_df.iloc[i]['Series_Title']
                        content_scores[title] = float(sim_score)
    elif genre:
        genre_corpus = merged_df[genre_col].fillna('').tolist()
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(genre_corpus)
        query_vector = tfidf.transform([genre])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        for i, sim_score in enumerate(similarities):
            title = merged_df.iloc[i]['Series_Title']
            content_scores[title] = float(sim_score)
    
    content_scores = normalize_scores(content_scores)
    
    # 2. Popularity scores
    popularity_scores = {}
    try:
        votes_col = 'No_of_Votes' if 'No_of_Votes' in merged_df.columns else 'Votes'
        for _, movie in merged_df.iterrows():
            title = movie['Series_Title']
            rating = movie.get(rating_col, 7.0)
            votes = movie.get(votes_col, 1000)
            
            if pd.isna(rating):
                rating = 7.0
            
            try:
                if isinstance(votes, str):
                    votes_val = float(votes.replace(',', ''))
                else:
                    votes_val = float(votes) if pd.notna(votes) else 1000.0
            except:
                votes_val = 1000.0
            
            normalized_rating = float(rating) / 10.0
            log_votes = np.log10(max(votes_val, 1.0))
            popularity = (normalized_rating * 0.7) + (min(log_votes / 6.0, 1.0) * 0.3)
            popularity_scores[title] = float(np.clip(popularity, 0.0, 1.0))
    except:
        for _, movie in merged_df.iterrows():
            title = movie['Series_Title']
            rating = movie.get(rating_col, 7.0)
            if pd.isna(rating):
                rating = 7.0
            popularity_scores[title] = float(rating) / 10.0
    
    # 3. Recency scores
    recency_scores = {}
    try:
        current_year = pd.Timestamp.now().year
        year_col = 'Released_Year' if 'Released_Year' in merged_df.columns else 'Year'
        for _, movie in merged_df.iterrows():
            title = movie['Series_Title']
            year = movie.get(year_col, 2000)
            
            try:
                if isinstance(year, str):
                    year_val = int(year.split()[0].strip('()'))
                else:
                    year_val = int(year) if pd.notna(year) else 2000
            except:
                year_val = 2000
            
            years_ago = max(0, current_year - year_val)
            recency = np.exp(-years_ago / 15.0)
            recency_scores[title] = float(np.clip(recency, 0.0, 1.0))
    except:
        for _, movie in merged_df.iterrows():
            recency_scores[movie['Series_Title']] = 0.5
    
    # Combine all candidates
    all_candidates = set()
    all_candidates.update(content_scores.keys())
    all_candidates.update(popularity_scores.keys())
    all_candidates.update(recency_scores.keys())
    
    # Remove target movie from candidates
    if target_movie and target_movie in all_candidates:
        all_candidates.remove(target_movie)
    
    # Calculate final scores
    final_scores = {}
    for title in all_candidates:
        content_score = content_scores.get(title, 0.0)
        popularity_score = popularity_scores.get(title, 0.0)
        recency_score = recency_scores.get(title, 0.0)
        
        # Apply the formula: FinalScore = content_weight×Content + popularity_weight×Popularity + recency_weight×Recency
        final_score = (content_weight * content_score + 
                      popularity_weight * popularity_score + 
                      recency_weight * recency_score)
        
        final_scores[title] = float(final_score)
    
    # Sort by final score and get top recommendations
    if not final_scores:
        return None
        
    sorted_recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    top_titles = [title for title, score in sorted_recommendations[:top_n]]
    
    if not top_titles:
        return None
    
    # Create result dataframe
    result_rows = []
    for title in top_titles:
        movie_data = merged_df[merged_df['Series_Title'] == title]
        if not movie_data.empty:
            result_rows.append(movie_data.iloc[0])
    
    if not result_rows:
        return None
    
    result_df = pd.DataFrame(result_rows)
    return result_df[['Series_Title', genre_col, rating_col]]