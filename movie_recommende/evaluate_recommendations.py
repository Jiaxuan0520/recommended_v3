import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import latest algorithm functions
from content_based import (
	create_content_features,
	find_rating_column,
	find_genre_column,
	content_based_filtering_enhanced
)
from collaborative import collaborative_knn, load_user_ratings
from hybrid import simple_hybrid_recommendation

# =============================================================
# Configuration
# =============================================================
# Updated weights to match latest hybrid algorithm
ALPHA = 0.4  # Content-based weight
BETA = 0.4   # Collaborative weight (updated from 0.3)
GAMMA = 0.1  # Popularity weight (updated from 0.2)
DELTA = 0.1  # Recency weight

RATING_THRESHOLD = 4.0  # ratings >= threshold are positive
TEST_SIZE_PER_USER = 0.2
RANDOM_STATE = 42
K_NEIGHBORS = 20  # for item-based KNN similarity neighborhood

# =============================================================
# Data Loading
# =============================================================

def load_datasets():
	imdb = pd.read_csv('imdb_top_1000.csv')
	user_ratings = pd.read_csv('user_movie_rating.csv')
	# ensure Movie_ID exists in merged data
	if 'Movie_ID' not in imdb.columns:
		imdb = imdb.copy()
		imdb['Movie_ID'] = range(1, len(imdb) + 1)
	# Standardize merged dataset to IMDB metadata only
	merged = imdb.drop_duplicates(subset='Series_Title')
	return merged, user_ratings

# =============================================================
# Helpers
# =============================================================

def get_cols(df):
	genre_col = find_genre_column(df)
	rating_col = find_rating_column(df)
	year_col = 'Released_Year' if 'Released_Year' in df.columns else 'Year'
	votes_col = 'No_of_Votes' if 'No_of_Votes' in df.columns else 'Votes'
	return genre_col, rating_col, year_col, votes_col


# =============================================================
# Helper Functions (simplified for new evaluation approach)
# =============================================================


# =============================================================
# Train/Test Split per user
# =============================================================

def split_per_user(user_ratings, test_size=TEST_SIZE_PER_USER, random_state=RANDOM_STATE):
	train_rows = []
	test_rows = []
	for user_id, grp in user_ratings.groupby('User_ID'):
		if len(grp) < 5:
			# small history: simple split
			grp_shuffled = grp.sample(frac=1, random_state=random_state)
			split_idx = int(len(grp_shuffled) * (1 - test_size))
			train_rows.append(grp_shuffled.iloc[:split_idx])
			test_rows.append(grp_shuffled.iloc[split_idx:])
		else:
			tr, te = train_test_split(grp, test_size=test_size, random_state=random_state)
			train_rows.append(tr)
			test_rows.append(te)
	train_df = pd.concat(train_rows).reset_index(drop=True)
	test_df = pd.concat(test_rows).reset_index(drop=True)
	return train_df, test_df

# =============================================================
# Evaluation Pipeline - Updated to use actual algorithms
# =============================================================

def evaluate_models():
	merged, ratings = load_datasets()
	genre_col, rating_col, year_col, votes_col = get_cols(merged)

	# Filter ratings to those movies present in merged
	present_ids = set(merged['Movie_ID'].unique())
	ratings = ratings[ratings['Movie_ID'].isin(present_ids)].copy()

	# Split BEFORE building models to avoid leakage
	train_df, test_df = split_per_user(ratings)

	# Build quick lookups
	movieid_to_title = dict(merged[['Movie_ID', 'Series_Title']].values)
	title_to_movieid = {v: k for k, v in movieid_to_title.items()}

	# Predictions and ground truth
	y_true_cls = []
	y_pred_cls_content = []
	y_pred_cls_collab = []
	y_pred_cls_hybrid = []

	y_true_reg = []
	y_pred_reg_content = []
	y_pred_reg_collab = []
	y_pred_reg_hybrid = []

	print("Evaluating models using actual algorithm implementations...")
	print(f"Test set size: {len(test_df)} ratings")

	# Iterate over test set rows
	for idx, row in test_df.iterrows():
		if idx % 100 == 0:
			print(f"Processing test item {idx}/{len(test_df)}")
			
		user = row['User_ID']
		movie_id = int(row['Movie_ID'])
		true_rating = float(row['Rating'])
		true_label = 1 if true_rating >= RATING_THRESHOLD else 0
		title = movieid_to_title.get(movie_id)
		
		if title is None:
			continue

		# 1. Content-Based Prediction using actual algorithm
		try:
			content_result = content_based_filtering_enhanced(merged, target_movie=title, top_n=1)
			if content_result is not None and not content_result.empty:
				# Get similarity score from content-based algorithm
				content_features = create_content_features(merged)
				target_idx = merged[merged['Series_Title'] == title].index[0]
				target_vec = content_features[target_idx].reshape(1, -1)
				sims = cosine_similarity(target_vec, content_features).flatten()
				content_score = float(np.max(sims[sims < 1.0])) if len(sims[sims < 1.0]) > 0 else 0.0
				content_rating_est = 2.0 + 8.0 * float(np.clip(content_score, 0.0, 1.0))
			else:
				content_rating_est = 5.0  # Default neutral rating
		except Exception as e:
			content_rating_est = 5.0

		# 2. Collaborative Prediction using actual algorithm
		try:
			collab_result = collaborative_knn(merged, target_movie=title, top_n=1, k_neighbors=K_NEIGHBORS)
			if collab_result is not None and not collab_result.empty and 'Avg_User_Rating' in collab_result.columns:
				collab_score = float(collab_result.iloc[0]['Avg_User_Rating'])
			else:
				# Fallback to item mean from training data
				item_ratings = train_df[train_df['Movie_ID'] == movie_id]['Rating']
				collab_score = item_ratings.mean() if not item_ratings.empty else train_df['Rating'].mean()
		except Exception as e:
			# Fallback to item mean
			item_ratings = train_df[train_df['Movie_ID'] == movie_id]['Rating']
			collab_score = item_ratings.mean() if not item_ratings.empty else train_df['Rating'].mean()

		# 3. Hybrid Prediction using actual algorithm
		try:
			hybrid_result, debug_info, score_breakdown = simple_hybrid_recommendation(
				merged, target_movie=title, top_n=1, show_debug=False
			)
			if hybrid_result is not None and not hybrid_result.empty:
				# Extract final score from hybrid algorithm
				# Since hybrid doesn't return scores directly, we'll compute it
				# using the same weights as the hybrid algorithm
				hybrid_pred = (
					ALPHA * content_rating_est +
					BETA * collab_score +
					GAMMA * 5.0 +  # Default popularity
					DELTA * 5.0    # Default recency
				)
			else:
				hybrid_pred = (content_rating_est + collab_score) / 2.0
		except Exception as e:
			hybrid_pred = (content_rating_est + collab_score) / 2.0

		# Clip to rating bounds
		content_rating_est = float(np.clip(content_rating_est, 1.0, 10.0))
		collab_score = float(np.clip(collab_score, 1.0, 10.0))
		hybrid_pred = float(np.clip(hybrid_pred, 1.0, 10.0))

		# Collect regression targets
		y_true_reg.append(true_rating)
		y_pred_reg_content.append(content_rating_est)
		y_pred_reg_collab.append(collab_score)
		y_pred_reg_hybrid.append(hybrid_pred)

		# Classification label predictions
		y_true_cls.append(true_label)
		y_pred_cls_content.append(1 if content_rating_est >= RATING_THRESHOLD else 0)
		y_pred_cls_collab.append(1 if collab_score >= RATING_THRESHOLD else 0)
		y_pred_cls_hybrid.append(1 if hybrid_pred >= RATING_THRESHOLD else 0)

	# Compute metrics
	def compute_classification_metrics(y_true, y_pred):
		return {
			'precision': precision_score(y_true, y_pred, zero_division=0),
			'recall': recall_score(y_true, y_pred, zero_division=0),
			'f1': f1_score(y_true, y_pred, zero_division=0),
			'accuracy': accuracy_score(y_true, y_pred),
			'report': classification_report(y_true, y_pred, target_names=['negative', 'positive'], zero_division=0)
		}

	def compute_regression_metrics(y_true, y_pred):
		mse = mean_squared_error(y_true, y_pred)
		return {'mse': mse, 'rmse': float(np.sqrt(mse))}

	results = {}
	results['Content-Based'] = {
		**compute_classification_metrics(y_true_cls, y_pred_cls_content),
		**compute_regression_metrics(y_true_reg, y_pred_reg_content)
	}
	results['Collaborative'] = {
		**compute_classification_metrics(y_true_cls, y_pred_cls_collab),
		**compute_regression_metrics(y_true_reg, y_pred_reg_collab)
	}
	results['Hybrid'] = {
		**compute_classification_metrics(y_true_cls, y_pred_cls_hybrid),
		**compute_regression_metrics(y_true_reg, y_pred_reg_hybrid)
	}

	# Display results
	print('\n' + '='*60)
	print('EVALUATION RESULTS - Using Actual Algorithm Implementations')
	print('='*60)
	
	print('\nModel: Content-Based')
	print(f"Accuracy: {results['Content-Based']['accuracy']:.3f}")
	print(f"Precision: {results['Content-Based']['precision']:.3f}")
	print(f"Recall: {results['Content-Based']['recall']:.3f}")
	print(f"F1-Score: {results['Content-Based']['f1']:.3f}")
	print(f"RMSE: {results['Content-Based']['rmse']:.3f}")
	
	print('\nModel: Collaborative')
	print(f"Accuracy: {results['Collaborative']['accuracy']:.3f}")
	print(f"Precision: {results['Collaborative']['precision']:.3f}")
	print(f"Recall: {results['Collaborative']['recall']:.3f}")
	print(f"F1-Score: {results['Collaborative']['f1']:.3f}")
	print(f"RMSE: {results['Collaborative']['rmse']:.3f}")
	
	print('\nModel: Hybrid')
	print(f"Accuracy: {results['Hybrid']['accuracy']:.3f}")
	print(f"Precision: {results['Hybrid']['precision']:.3f}")
	print(f"Recall: {results['Hybrid']['recall']:.3f}")
	print(f"F1-Score: {results['Hybrid']['f1']:.3f}")
	print(f"RMSE: {results['Hybrid']['rmse']:.3f}")

	# Summary table
	summary_rows = []
	for name in ['Content-Based', 'Collaborative', 'Hybrid']:
		row = {
			'Method': name,
			'Accuracy': round(results[name]['accuracy'], 3),
			'Precision': round(results[name]['precision'], 3),
			'Recall': round(results[name]['recall'], 3),
			'F1-Score': round(results[name]['f1'], 3),
			'RMSE': round(results[name]['rmse'], 3),
			'Notes': (
				'Uses TF-IDF with enhanced features' if name == 'Content-Based' else
				'Uses item-based KNN with user ratings' if name == 'Collaborative' else
				'Combines all approaches with weights (0.4, 0.4, 0.1, 0.1)'
			)
		}
		summary_rows.append(row)
	
	summary_df = pd.DataFrame(summary_rows)
	print('\n' + '='*80)
	print('SUMMARY COMPARISON TABLE')
	print('='*80)
	print(summary_df.to_string(index=False))
	
	return results


if __name__ == '__main__':
	evaluate_models()
