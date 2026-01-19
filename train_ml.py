import pandas as pd
import pickle
import os
import logging
import subprocess
import mlflow
import mlflow.sklearn
from datetime import datetime
from sqlalchemy import create_engine, text
from collections import defaultdict

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
if os.path.exists('/.dockerenv') or os.path.exists('/opt/airflow'):
    DB_URL = "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow"
    MODEL_DIR = "/opt/airflow/models"
    LAKE_ROOT = "/opt/airflow/data_lake"
    MLFLOW_URI = "http://mlflow:5000"
else:
    DB_URL = "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"
    MODEL_DIR = "models"
    LAKE_ROOT = "data_lake"
    MLFLOW_URI = "http://172.17.0.1:5000"

engine = create_engine(DB_URL)
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("RecoMart_V3_System")

# --- LINEAGE HELPER: FULFILLS DVC REQUIREMENT ---
def get_dvc_hash():
    """Versions raw and transformed data using DVC (Requirement 1)."""
    try:
        # Runs 'dvc add' to ensure versioning is current
        subprocess.run(['dvc', 'add', LAKE_ROOT], check=True, capture_output=True)
        with open(f"{LAKE_ROOT}.dvc", 'r') as f:
            import yaml
            dvc_meta = yaml.safe_load(f)
            # The 8-char hash is the unique lineage ID
            return dvc_meta['outs'][0]['md5'][:8]
    except Exception as e:
        logger.warning(f"Lineage Warning: Could not resolve DVC hash ({e})")
        return "manual_dev"

# --- ALIGNED HEALTH LOGGER ---
def log_model_health(metrics, run_id=None, data_hash=None):
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO model_health_logs (rmse, precision, recall, f1_score, model_version, mlflow_run_id, data_hash)
                VALUES (:rmse, :p, :r, :f1, :ver, :run_id, :hash)
            """), {
                "rmse": metrics.get('rmse', 0.0),
                "p": metrics['precision'],
                "r": metrics['recall'],
                "f1": metrics.get('f1', 0.0),
                "ver": f"svd_{datetime.now().strftime('%m%d_%H%M')}",
                "run_id": run_id,
                "hash": data_hash
            })
    except Exception as e:
        logger.warning(f"⚠️ Health logging failed: {e}")

def precision_recall_at_k(predictions, k=5, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    precisions, recalls = {}, {}
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
    return sum(precisions.values()) / len(precisions) if precisions else 0, \
           sum(recalls.values()) / len(recalls) if recalls else 0

def train_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    current_data_hash = get_dvc_hash() 

    from surprise import SVD, Dataset, Reader, dump, accuracy
    from surprise.model_selection import train_test_split

    logger.info("Starting High-Visibility Lineage Run...")
    try:
        df_ratings = pd.read_sql("SELECT user_id, product_id, rating FROM temp_lake_ratings_raw", engine)

        if not df_ratings.empty:
            # Generate a distinct run name for easier UI spotting
            with mlflow.start_run(run_name=f"LINEAGE_FIX_{datetime.now().strftime('%H%M%S')}") as run:
                
                # --- TRIPLE-REDUNDANCY LOGGING TO FIX UI HYPHENS ---
                # Requirement: Track Metadata (Source, Date, Transformations)
                metadata = {
                    "ingestion_date": datetime.now().strftime("%Y-%m-%d"),
                    "data_source": "temp_lake_ratings_raw",
                    "transformations": "Vader-Sentiment, Rating-Clipping, DVC-Hashing",
                    "data_hash": current_data_hash,
                    "dvc_hash": current_data_hash
                }

                # Log every field as BOTH a parameter and a tag
                for key, val in metadata.items():
                    mlflow.log_param(key, val)   # Populates 'Parameters' columns
                    mlflow.set_tag(key, val)     # Populates 'Tags' columns

                # --- COLLABORATIVE FILTERING TRAINING ---
                reader = Reader(rating_scale=(1, 5))
                data = Dataset.load_from_df(df_ratings[['user_id', 'product_id', 'rating']], reader)
                trainset, testset = train_test_split(data, test_size=0.2)
                
                algo = SVD()
                algo.fit(trainset)
                predictions = algo.test(testset)

                # Metrics
                rmse_val = accuracy.rmse(predictions)
                p, r = precision_recall_at_k(predictions, k=5, threshold=3.5)
                f1_val = (2 * p * r) / (p + r) if (p + r) > 0 else 0
                metrics = {"rmse": round(float(rmse_val), 4), "precision": round(float(p), 4), 
                           "recall": round(float(r), 4), "f1": round(float(f1_val), 4)}

                # Log standard MLflow components
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(algo, "svd-model")
                mlflow.register_model(f"runs:/{run.info.run_id}/svd-model", "RecoMart_SVD_Model")
                
                # Sync with Database for Lineage Documentation
                log_model_health(metrics, run_id=run.info.run_id, data_hash=current_data_hash)

                dump.dump(os.path.join(MODEL_DIR, "svd_v1.pkl"), algo=algo)
                logger.info(f"✅ Lineage Captured in UI. Hash: {current_data_hash}")
        else:
            logger.warning("⚠️ No data found in temp_lake_ratings_raw.")
    except Exception as e:
        logger.error(f"❌ Error during SVD training: {e}")

if __name__ == "__main__":
    train_models()
