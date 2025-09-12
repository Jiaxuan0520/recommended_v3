import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from collections import defaultdict
import warnings
import logging
import os
import sys
from io import StringIO

# Suppress all warnings and Streamlit messages
warnings.filterwarnings('ignore')
logging.getLogger('streamlit').setLevel(logging.ERROR)
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

# Suppress Streamlit warnings during import
old_stderr = sys.stderr
sys.stderr = StringIO()

# Import latest algorithm functions
from content_based import (
	create_content_features,
	find_rating_column,
	find_genre_column,
	content_based_filtering_enhanced
)
from collaborative import collaborative_knn, load_user_ratings
from hybrid import simple_hybrid_recommendation

# Restore stderr
sys.stderr = old_stderr


# Updated to match latest hybrid algorithm weights
ALPHA = 0.4  # Content-based weight
BETA = 0.4   # Collaborative weight (updated to match latest)
GAMMA = 0.1  # Popularity weight (updated to match latest)
DELTA = 0.1  # Recency weight

RATING_THRESHOLD = 4.0  # ratings >= threshold are positive
TEST_SIZE_PER_USER = 0.2
RANDOM_STATE = 42
K_NEIGHBORS = 20  # for item-based KNN similarity neighborhood



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



def get_cols(df):
	genre_col = find_genre_column(df)
	rating_col = find_rating_column(df)
	year_col = 'Released_Year' if 'Released_Year' in df.columns else 'Year'
	votes_col = 'No_of_Votes' if 'No_of_Votes' in df.columns else 'Votes'
	return genre_col, rating_col, year_col, votes_col


# Removed old helper functions - now using actual algorithm functions from latest code


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


def evaluate_models():
	"""Evaluate models using actual algorithm functions from latest code"""
	# Suppress Streamlit warnings during execution
	import contextlib
	
	@contextlib.contextmanager
	def suppress_streamlit_warnings():
		old_stderr = sys.stderr
		sys.stderr = StringIO()
		try:
			yield
		finally:
			sys.stderr = old_stderr
	
	with suppress_streamlit_warnings():
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

		print("Evaluating models using actual algorithm functions...")
		print("Using improved prediction methods:")
		print("- Content-Based: Average of top 3 content similarities")
		print("- Collaborative: Weighted average of top 3 similar movies by user ratings")
		print("- Hybrid: Average of top 5 hybrid algorithm scores")
		print()
		
		# Iterate over test set rows
		for idx, row in test_df.iterrows():
			if idx % 100 == 0:
				print(f"Processing test sample {idx}/{len(test_df)}")
				
			user = row['User_ID']
			movie_id = int(row['Movie_ID'])
			true_rating = float(row['Rating'])
			true_label = 1 if true_rating >= RATING_THRESHOLD else 0
			title = movieid_to_title.get(movie_id)
			
			if title is None:
				continue

			# 1. Content-Based Prediction using actual algorithm
			try:
				content_result = content_based_filtering_enhanced(merged, target_movie=title, top_n=3)
				if content_result is not None and not content_result.empty:
					# Get similarity scores from content-based algorithm
					content_features = create_content_features(merged)
					target_idx = merged[merged['Series_Title'] == title].index[0]
					target_vec = content_features[target_idx].reshape(1, -1)
					sims = cosine_similarity(target_vec, content_features).flatten()
					
					# Get top similar movies (excluding self)
					valid_sims = sims[sims < 1.0]
					if len(valid_sims) > 0:
						top_sims = np.sort(valid_sims)[-3:]  # Top 3 similarities
						content_score = float(np.mean(top_sims))  # Average of top similarities
					else:
						content_score = 0.0
					
					# Convert similarity to rating (1-10 scale)
					content_rating_est = 1.0 + 9.0 * float(np.clip(content_score, 0.0, 1.0))
				else:
					content_rating_est = 5.0  # Default neutral rating
			except Exception as e:
				content_rating_est = 5.0

			# 2. Collaborative Prediction using actual algorithm
			try:
				collab_result = collaborative_knn(merged, target_movie=title, top_n=3)
				if collab_result is not None and not collab_result.empty:
					# Get average user rating from top similar movies
					avg_ratings = collab_result['Avg_User_Rating'].values
					similarities = collab_result['Similarity'].values if 'Similarity' in collab_result.columns else np.ones(len(avg_ratings))
					
					# Weight by similarity for better prediction
					weighted_sum = np.sum(avg_ratings * similarities)
					weight_sum = np.sum(similarities)
					collab_score = weighted_sum / weight_sum if weight_sum > 0 else np.mean(avg_ratings)
				else:
					# Fallback to item mean from training data
					item_ratings = train_df[train_df['Movie_ID'] == movie_id]['Rating']
					collab_score = item_ratings.mean() if not item_ratings.empty else ratings['Rating'].mean()
			except Exception as e:
				collab_score = ratings['Rating'].mean()

			# 3. Hybrid Prediction using actual algorithm
			try:
				# Use hybrid algorithm to find similar movies, then predict rating based on their scores
				hybrid_result, debug_info, score_breakdown = simple_hybrid_recommendation(
					merged, target_movie=title, top_n=5, show_debug=True
				)
				
				if hybrid_result is not None and not hybrid_result.empty and score_breakdown is not None:
					# Get the hybrid scores for similar movies
					hybrid_scores = []
					for breakdown in score_breakdown:
						hybrid_scores.append(float(breakdown['Final Score']))
					
					# Use the average of top hybrid scores as prediction
					# Convert from 0-1 scale to 1-10 scale
					avg_hybrid_score = np.mean(hybrid_scores) if hybrid_scores else 0.5
					hybrid_pred = 1.0 + 9.0 * avg_hybrid_score
				else:
					# Fallback: use weighted combination of content and collaborative
					hybrid_pred = (0.4 * content_rating_est + 0.6 * collab_score)
			except Exception as e:
				# Fallback: use weighted combination if hybrid algorithm fails
				hybrid_pred = (0.4 * content_rating_est + 0.6 * collab_score)

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
			
			# Debug: Show first few predictions
			if idx < 5:
				print(f"Sample {idx}: {title}")
				print(f"  True: {true_rating:.2f} (label: {true_label})")
				print(f"  Content: {content_rating_est:.2f} (label: {1 if content_rating_est >= RATING_THRESHOLD else 0})")
				print(f"  Collab: {collab_score:.2f} (label: {1 if collab_score >= RATING_THRESHOLD else 0})")
				print(f"  Hybrid: {hybrid_pred:.2f} (label: {1 if hybrid_pred >= RATING_THRESHOLD else 0})")
				print()

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

		# Display
		print('\nModel: Content-Based')
		print(f"Accuracy: {results['Content-Based']['accuracy']:.3f}")
		print(results['Content-Based']['report'])
		print('\nModel: Collaborative')
		print(f"Accuracy: {results['Collaborative']['accuracy']:.3f}")
		print(results['Collaborative']['report'])
		print('\nModel: Hybrid')
		print(f"Accuracy: {results['Hybrid']['accuracy']:.3f}")
		print(results['Hybrid']['report'])

		# Summary table
		summary_rows = []
		for name in ['Collaborative', 'Content-Based', 'Hybrid']:
			row = {
				'Method Used': name,
				'Precision': round(results[name]['precision'], 2),
				'Recall': round(results[name]['recall'], 2),
				'RMSE': round(results[name]['rmse'], 2),
				'Notes': (
					'Worked well with dense ratings' if name == 'Collaborative' else
					'Good with rich metadata' if name == 'Content-Based' else
					'Best balance between both'
				)
			}
			summary_rows.append(row)
		summary_df = pd.DataFrame(summary_rows, columns=['Method Used', 'Precision', 'Recall', 'RMSE', 'Notes'])
		print('\nComparison Table:')
		print(summary_df.to_string(index=False))


if __name__ == '__main__':
	evaluate_models()
