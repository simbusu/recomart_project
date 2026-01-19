import os
import json
import glob
import logging
import time
import pandas as pd
import mlflow
import numpy as np
import subprocess
import yaml
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from sqlalchemy import text, create_engine
import sys

# Add the project root to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from init_db import setup_database
from ingester import RecoMartETL

# --- CONFIGURATION ---
logger = logging.getLogger("airflow.task")

if os.path.exists('/.dockerenv') or os.path.exists('/opt/airflow'):
    DB_URL = "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow"
    LAKE_ROOT = "/opt/airflow/data_lake"
    NLTK_DATA_DIR = "/opt/airflow/nltk_data"
    MODEL_DIR = "/opt/airflow/models"
    MLFLOW_URI = "http://mlflow:5000"
else:
    DB_URL = "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"
    LAKE_ROOT = os.path.expanduser("~/recomart_project/data_lake")
    NLTK_DATA_DIR = os.path.expanduser("~/recomart_project/nltk_data")
    MODEL_DIR = os.path.expanduser("~/recomart_project/models")
    MLFLOW_URI = "http://172.17.0.1:5000"

os.makedirs(MODEL_DIR, exist_ok=True)

default_args = {
    'owner': 'recomart_admin',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# --- DASHBOARD BRIDGE HELPER ---
def log_model_health_to_db(metrics, run_id=None, data_hash=None):
    """Writes training results to DB for dashboard tracking."""
    try:
        engine_db = create_engine(DB_URL)
        with engine_db.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS model_health_logs (
                    id SERIAL PRIMARY KEY,
                    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    rmse REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    model_version VARCHAR(50),
                    mlflow_run_id TEXT,
                    data_hash TEXT
                )
            """))
            conn.execute(text("""
                INSERT INTO model_health_logs (rmse, precision, recall, f1_score, model_version, mlflow_run_id, data_hash)
                VALUES (:rmse, :p, :r, :f1, :ver, :run_id, :hash)
            """), {
                "rmse": metrics.get('rmse', 0.0),
                "p": metrics.get('precision', 0.0),
                "r": metrics.get('recall', 0.0),
                "f1": metrics.get('f1', 0.0),
                "ver": f"svd_v3_{datetime.now().strftime('%m%d_%H%M')}",
                "run_id": run_id,
                "hash": data_hash
            })
    except Exception as e:
        logger.warning(f"⚠️ Dashboard logging failed: {e}")

# --- 1. INGESTION & HOUSEKEEPING ---
def run_user_sync():
    etl = RecoMartETL()
    etl.sync_users_csv()

def run_product_sync():
    etl = RecoMartETL()
    api_url = 'http://host.docker.internal:5001/api/products' if os.path.exists('/.dockerenv') else 'http://localhost:5001/api/products'
    headers = {"X-API-KEY": "recomart_secret_2026"}
    with etl.engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS products_temp;"))
    etl.sync_products_api(api_url, headers=headers)

def cleanup_data_lake():
    now = time.time()
    retention_period = 2 * 24 * 60 * 60
    count = 0
    for root, dirs, files in os.walk(LAKE_ROOT):
        for name in files:
            file_path = os.path.join(root, name)
            if os.stat(file_path).st_mtime < (now - retention_period):
                os.remove(file_path)
                count += 1
    logger.info(f"✅ Cleanup Complete: Removed {count} old lake files.")

# --- 2. VALIDATION ---
def validate_lake_data(**kwargs):
    search_path = os.path.join(LAKE_ROOT, "kafka/**/*.jsonl")
    found_files = glob.glob(search_path, recursive=True)
    if not found_files:
        raise ValueError("❌ Validation Failed: No event data found in lake.")
    logger.info(f"✅ Validation Passed: Found {len(found_files)} files.")

# --- 3. PREPARATION & SENTIMENT (DVC VERSIONING) ---
def prepare_and_stage_data(**kwargs):
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(NLTK_DATA_DIR, 'sentiment/vader_lexicon.zip')):
        nltk.download('vader_lexicon', download_dir=NLTK_DATA_DIR)
    nltk.data.path.append(NLTK_DATA_DIR)

    all_reviews = []
    search_path = os.path.join(LAKE_ROOT, "kafka/product_review/**/*.jsonl")
    sid = SentimentIntensityAnalyzer()

    for file_path in glob.glob(search_path, recursive=True):
        with open(file_path, 'r') as f:
            for line in f:
                try: all_reviews.append(json.loads(line))
                except: continue

    if all_reviews:
        df = pd.DataFrame(all_reviews)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(3.0).clip(1.0, 5.0)
        df['sentiment'] = df['review_text'].apply(lambda x: sid.polarity_scores(str(x))['compound'] if x else 0.0)
        df = df.drop_duplicates(subset=['user_id', 'product_id'], keep='last')

        # DVC VERSIONING [Requirement 8]
        versioned_csv = os.path.join(LAKE_ROOT, "transformed_ratings.csv")
        df.to_csv(versioned_csv, index=False)

        data_hash = "unknown_hash"
        try:
            dvc_bin = os.path.join(os.path.dirname(sys.executable), 'dvc')
            subprocess.run([dvc_bin, "add", LAKE_ROOT], check=True, capture_output=True)
            dvc_pointer = os.path.join(os.path.dirname(LAKE_ROOT), "data_lake.dvc")

            if os.path.exists(dvc_pointer):
                with open(dvc_pointer, "r") as f:
                    dvc_config = yaml.safe_load(f)
                    data_hash = dvc_config['outs'][0]['md5']
                logger.info(f"✅ DVC Versioning Successful: {data_hash}")
        except Exception as e:
            logger.error(f"⚠️ DVC Error: {str(e)}")

        kwargs['ti'].xcom_push(key='dvc_data_hash', value=data_hash)

        engine = create_engine(DB_URL)
        df[['product_id', 'user_id', 'rating', 'sentiment']].to_sql(
            'temp_lake_ratings_raw', engine, if_exists='replace', index=False
        )
    else:
        logger.warning("No reviews found to stage.")

# --- 4. ML TRAINING & METADATA LINEAGE ---
def train_collaborative_model(**kwargs):
    from surprise import SVD, Dataset, Reader, accuracy
    from surprise.model_selection import train_test_split
    from mlflow.tracking import MlflowClient
    import mlflow.sklearn

    ti = kwargs['ti']
    dvc_hash = ti.xcom_pull(key='dvc_data_hash', task_ids='process_sentiment_scores') or "unknown_hash"

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("RecoMart_V3_System")
    engine = create_engine(DB_URL)

    try:
        df = pd.read_sql("SELECT user_id, product_id, rating FROM temp_lake_ratings_raw", engine)

        if len(df) > 5:
            run_name = f"SVD_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with mlflow.start_run(run_name=run_name) as run:
                # --- REQUIREMENT 8: LINEAGE ---
                lineage_meta = {
                    "data_source": "Kafka-Data-Lake/Transformed-CSV",
                    "ingestion_date": datetime.now().strftime("%Y-%m-%d"),
                    "transformations": "Vader-Sentiment, Rating-Clipping, Deduplication",
                    "dvc_hash": dvc_hash,
                    "data_hash": dvc_hash
                }
                for k, v in lineage_meta.items():
                    mlflow.set_tag(k, v)
                    mlflow.log_param(k, v)

                reader = Reader(rating_scale=(1, 5))
                data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)

                # Split for evaluation
                trainset, testset = train_test_split(data, test_size=0.2)
                algo = SVD()
                algo.fit(trainset)
                predictions = algo.test(testset)
                
                # --- CALCULATE METRICS ---
                rmse_val = accuracy.rmse(predictions, verbose=False)
                
                # Logic for Precision@K and Recall@K
                user_est_true = {}
                for uid, _, true_r, est, _ in predictions:
                    user_est_true.setdefault(uid, []).append((est, true_r))

                precisions = []
                recalls = []
                for uid, user_ratings in user_est_true.items():
                    user_ratings.sort(key=lambda x: x[0], reverse=True)
                    n_rel = sum((true_r >= 3.5) for (_, true_r) in user_ratings)
                    n_rec_k = sum((est >= 3.5) for (est, _) in user_ratings[:5])
                    n_rel_and_rec_k = sum(((true_r >= 3.5) and (est >= 3.5)) for (est, true_r) in user_ratings[:5])
                    
                    precisions.append(n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0)
                    recalls.append(n_rel_and_rec_k / n_rel if n_rel != 0 else 0)

                prec_val = np.mean(precisions)
                rec_val = np.mean(recalls)
                f1_val = (2 * prec_val * rec_val) / (prec_val + rec_val) if (prec_val + rec_val) > 0 else 0

                # Log metrics to MLflow
                metrics_dict = {
                    "rmse": rmse_val,
                    "precision": prec_val,
                    "recall": rec_val,
                    "f1": f1_val
                }
                mlflow.log_metrics(metrics_dict)

                # Log and Register Model
                mlflow.sklearn.log_model(sk_model=algo, artifact_path="svd-model")
                model_name = "RecoMart_SVD_Model"
                mv = mlflow.register_model(f"runs:/{run.info.run_id}/svd-model", model_name)

                # Link Lineage to Model Version
                client = MlflowClient()
                client.set_model_version_tag(model_name, mv.version, "data_lineage", dvc_hash)

                # Log to DB for Dashboard
                log_model_health_to_db(metrics_dict, run_id=run.info.run_id, data_hash=dvc_hash)

                logger.info(f"✅ Model Registered with full metrics and Lineage: {dvc_hash}")

    except Exception as e:
        logger.error(f"❌ ML Training error: {str(e)}")

# --- 5. FEATURE STORE & SIMILARITY ---
def transform_to_feature_store():
    engine = create_engine(DB_URL)
    upsert_query = text("""
        INSERT INTO product_feature_store (product_id, total_sales, avg_sentiment, review_count, updated_at)
        SELECT p.product_id, COALESCE(pfs.total_sales, 0), COALESCE(lake.avg_sent, 0), COALESCE(lake.cnt, 0), NOW()
        FROM products p
        LEFT JOIN product_feature_store pfs ON p.product_id = pfs.product_id
        LEFT JOIN (
            SELECT product_id, AVG(sentiment) as avg_sent, COUNT(*) as cnt
            FROM temp_lake_ratings_raw GROUP BY product_id
        ) lake ON p.product_id = lake.product_id
        ON CONFLICT (product_id) DO UPDATE SET
            avg_sentiment = EXCLUDED.avg_sentiment, review_count = EXCLUDED.review_count, updated_at = NOW();
    """)
    with engine.begin() as conn:
        conn.execute(upsert_query)

def generate_content_similarity():
    import pickle
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    engine = create_engine(DB_URL)
    df_p = pd.read_sql("SELECT product_id, product_name FROM products", engine)
    if not df_p.empty:
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df_p['product_name'])
        sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        with open(os.path.join(MODEL_DIR, "content_sim.pkl"), "wb") as f:
            pickle.dump((sim_matrix, df_p), f)

# --- DAG DEFINITION ---
with DAG(
    'recomart_full_pipeline',
    default_args=default_args,
    schedule='@daily',
    catchup=False,
    tags=['recomart', 'mlops', 'lineage']
) as dag:

    initialize_database = PythonOperator(task_id='initialize_database', python_callable=setup_database)
    sync_users = PythonOperator(task_id='sync_users_csv', python_callable=run_user_sync)
    sync_products = PythonOperator(task_id='sync_products_api', python_callable=run_product_sync)
    cleanup_lake = PythonOperator(task_id='cleanup_old_lake_files', python_callable=cleanup_data_lake)
    validate_lake = PythonOperator(task_id='validate_lake_data', python_callable=validate_lake_data)
    prepare_sentiment = PythonOperator(task_id='process_sentiment_scores', python_callable=prepare_and_stage_data)
    train_ml_model = PythonOperator(task_id='train_svd_recommendation_model', python_callable=train_collaborative_model)
    update_feature_store = PythonOperator(task_id='update_product_feature_store', python_callable=transform_to_feature_store)
    content_sim_task = PythonOperator(task_id='generate_content_similarity', python_callable=generate_content_similarity)

    initialize_database >> [sync_users, sync_products, cleanup_lake] >> validate_lake >> \
    prepare_sentiment >> [train_ml_model, content_sim_task] >> update_feature_store
