import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from content_based import create_content_features, find_rating_column, find_genre_column
from collaborative import collaborative_knn, load_user_ratings
import warnings
warnings.filterwarnings('ignore')


def simple_hybrid_recommendation(merged_df, target_movie=None, genre=None, top_n=8, show_debug=False):
    """
    Simple hybrid recommendation without caching - returns (results, debug_info, score_breakdown)
    Formula: FinalScore = 0.4×Content + 0.4×Collaborative + 0.1×Popularity + 0.1×Recency
    """
    
    # Algorithm weights
    alpha = 0.4  # Content-based weight
    beta = 0.4   # Collaborative weight
    gamma = 0.1  # Popularity weight
    delta = 0.1  # Recency weight
    
    rating_col = find_rating_column(merged_df)
    genre_col = find_genre_column(merged_df)
    user_ratings_df = load_user_ratings()
    
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
        try:
            content_features = create_content_features(merged_df)
            if target_movie in merged_df['Series_Title'].values:
                target_idx = merged_df[merged_df['Series_Title'] == target_movie].index[0]
            else:
                match_series = merged_df[merged_df['Series_Title'].str.lower() == str(target_movie).lower()]
                if not match_series.empty:
                    target_idx = match_series.index[0]
                else:
                    target_idx = None
            
            if target_idx is not None:
                target_loc = merged_df.index.get_loc(target_idx)
                target_vec = content_features[target_loc].reshape(1, -1)
                sims = cosine_similarity(target_vec, content_features).flatten()
                for i, sim_score in enumerate(sims):
                    if i != target_loc:
                        title = merged_df.iloc[i]['Series_Title']
                        content_scores[title] = float(sim_score)
        except Exception:
            pass
    elif genre:
        try:
            genre_corpus = merged_df[genre_col].fillna('').tolist()
            tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = tfidf.fit_transform(genre_corpus)
            query_vector = tfidf.transform([genre])
            sims = cosine_similarity(query_vector, tfidf_matrix).flatten()
            for i, sim_score in enumerate(sims):
                title = merged_df.iloc[i]['Series_Title']
                content_scores[title] = float(sim_score)
        except Exception:
            pass
    
    content_scores = normalize_scores(content_scores)
    
    # 2. Collaborative scores
    collab_scores = {}
    if target_movie and user_ratings_df is not None:
        try:
            collab_results = collaborative_knn(merged_df, target_movie, top_n=top_n * 5, k_neighbors=50)
            if collab_results is not None and not collab_results.empty:
                if 'Similarity' in collab_results.columns:
                    for _, row in collab_results.iterrows():
                        title = row['Series_Title']
                        similarity = row.get('Similarity', 0.0)
                        if pd.notna(similarity):
                            collab_scores[title] = float(similarity)
        except Exception:
            pass
    
    collab_scores = normalize_scores(collab_scores)
    
    # 3. Popularity scores
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
    except Exception:
        for _, movie in merged_df.iterrows():
            title = movie['Series_Title']
            rating = movie.get(rating_col, 7.0)
            if pd.isna(rating):
                rating = 7.0
            popularity_scores[title] = float(rating) / 10.0
    
    # 4. Recency scores
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
    except Exception:
        for _, movie in merged_df.iterrows():
            recency_scores[movie['Series_Title']] = 0.5
    
    # Debug info
    debug_info = {
        'content_candidates': len(content_scores),
        'collab_candidates': len(collab_scores),
        'popularity_candidates': len(popularity_scores),
        'recency_candidates': len(recency_scores)
    }
    
    # Combine all candidates
    all_candidates = set()
    all_candidates.update(content_scores.keys())
    all_candidates.update(collab_scores.keys())
    all_candidates.update(popularity_scores.keys())
    all_candidates.update(recency_scores.keys())
    
    # Remove target movie from candidates
    if target_movie and target_movie in all_candidates:
        all_candidates.remove(target_movie)
    
    # Calculate final hybrid scores
    final_scores = {}
    for title in all_candidates:
        content_score = content_scores.get(title, 0.0)
        collab_score = collab_scores.get(title, 0.0)
        popularity_score = popularity_scores.get(title, 0.0)
        recency_score = recency_scores.get(title, 0.0)
        
        # Apply the formula: FinalScore = 0.4×Content + 0.4×Collaborative + 0.1×Popularity + 0.1×Recency
        final_score = (alpha * content_score + 
                      beta * collab_score + 
                      gamma * popularity_score + 
                      delta * recency_score)
        
        final_scores[title] = float(final_score)
    
    # Sort by final score and get top recommendations
    if not final_scores:
        return None, debug_info, None
        
    sorted_recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    top_titles = [title for title, score in sorted_recommendations[:top_n]]
    
    if not top_titles:
        return None, debug_info, None
    
    # Create result dataframe
    result_rows = []
    for title in top_titles:
        movie_data = merged_df[merged_df['Series_Title'] == title]
        if not movie_data.empty:
            result_rows.append(movie_data.iloc[0])
    
    if not result_rows:
        return None, debug_info, None
    
    result_df = pd.DataFrame(result_rows)
    
    # Prepare score breakdown if requested
    score_breakdown = None
    if show_debug:
        score_breakdown = []
        for title in top_titles:
            breakdown = {
                'Movie': title,
                'Content': f"{content_scores.get(title, 0.0):.3f}",
                'Collaborative': f"{collab_scores.get(title, 0.0):.3f}",
                'Popularity': f"{popularity_scores.get(title, 0.0):.3f}",
                'Recency': f"{recency_scores.get(title, 0.0):.3f}",
                'Final Score': f"{final_scores.get(title, 0.0):.3f}"
            }
            score_breakdown.append(breakdown)
    
    final_result = result_df[['Series_Title', genre_col, rating_col]]
    return final_result, debug_info, score_breakdown


# Remove the @st.cache_data decorator to avoid caching issues
def smart_hybrid_recommendation(merged_df, target_movie=None, genre=None, top_n=8, show_debug=False):
    """Non-cached wrapper for hybrid recommendation"""
    return simple_hybrid_recommendation(merged_df, target_movie, genre, top_n, show_debug)
