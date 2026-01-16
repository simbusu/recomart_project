import pandas as pd
import pickle
import os
import json
import logging
from datetime import datetime
from sqlalchemy import create_engine
from collections import defaultdict

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- DATABASE CONNECTION ---
# Automatically switch between Docker and Localhost networking
if os.path.exists('/.dockerenv') or os.path.exists('/opt/airflow'):
    DB_URL = "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow"
    MODEL_DIR = "/opt/airflow/models" # Ensuring Airflow path is used
else:
    DB_URL = "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"
    MODEL_DIR = "models"

engine = create_engine(DB_URL)

def precision_recall_at_k(predictions, k=5, threshold=3.5):
    """Calculates precision and recall at k for the dashboard metrics."""
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    avg_prec = sum(prec for prec in precisions.values()) / len(precisions) if precisions else 0
    avg_rec = sum(rec for rec in recalls.values()) / len(recalls) if recalls else 0
    return avg_prec, avg_rec

def train_models():
    # 1. Prepare Model Directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 2. Train Collaborative Filtering (SVD)
    from surprise import SVD, Dataset, Reader, dump
    from surprise.model_selection import train_test_split
    
    logger.info("Training SVD Model...")
    try:
        df_ratings = pd.read_sql("SELECT user_id, product_id, rating FROM temp_lake_ratings_raw", engine)

        if not df_ratings.empty:
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(df_ratings[['user_id', 'product_id', 'rating']], reader)
            
            # --- Added for Metrics ---
            trainset, testset = train_test_split(data, test_size=0.2)
            algo = SVD()
            algo.fit(trainset)
            predictions = algo.test(testset)
            
            p, r = precision_recall_at_k(predictions, k=5, threshold=3.5)
            
            # Save metrics to JSON for the Dashboard Metric Cards
            metrics = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(os.path.join(MODEL_DIR, "latest_metrics.json"), "w") as f:
                json.dump(metrics, f)
            # -------------------------

            # Save the SVD Model exactly as before
            dump.dump(os.path.join(MODEL_DIR, "svd_v1.pkl"), algo=algo)
            logger.info(f"✅ SVD Model Saved. Metrics: P={p:.2f}, R={r:.2f}")
        else:
            logger.warning("⚠️ No rating data found in temp_lake_ratings_raw. Skipping SVD.")
    except Exception as e:
        logger.error(f"❌ SVD Training Error: {e}")

    # 3. Train Content Filtering (TF-IDF)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    logger.info("Building Content Similarity Matrix...")
    try:
        df_prod = pd.read_sql("SELECT product_id, product_name, category FROM products", engine)

        if not df_prod.empty:
            # Added .fillna('') to category to prevent errors during metadata concat
            df_prod['metadata'] = df_prod['product_name'] + " " + df_prod['category'].fillna('')
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(df_prod['metadata'])
            sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

            # Saved exactly as before
            with open(os.path.join(MODEL_DIR, "content_sim.pkl"), "wb") as f:
                pickle.dump((sim_matrix, df_prod), f)
            logger.info("✅ Content Matrix Saved.")
        else:
            logger.warning("⚠️ No products found in database. Skipping Content Filtering.")
    except Exception as e:
        logger.error(f"❌ Content Matrix Error: {e}")

if __name__ == "__main__":
    train_models()
