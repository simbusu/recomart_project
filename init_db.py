import os
import logging
from sqlalchemy import create_engine, text

# --- LOGGING CONFIGURATION ---
log_file = "init_db.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def setup_database():
    """
    Initializes the database schema for the RecoMart project.
    Includes MLOps Lineage tracking for DVC and MLflow.
    """
    # Environment-based connection string
    if os.path.exists('/.dockerenv') or os.path.exists('/opt/airflow'):
        db_url = "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow"
    else:
        db_url = "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"

    logger.info(f"Connecting to database: {db_url}")
    engine = create_engine(db_url)

    # 1. Users table
    create_users = """
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        name TEXT,
        email TEXT,
        gender TEXT,
        city TEXT,
        signup_date DATE
    );"""

    # 2. Products table
    create_products = """
    CREATE TABLE IF NOT EXISTS products (
        product_id TEXT PRIMARY KEY,
        product_name TEXT,
        category TEXT,
        price REAL,
        rating REAL
    );"""

    # 3. Orders table
    create_orders = """
    CREATE TABLE IF NOT EXISTS orders (
        order_id TEXT PRIMARY KEY,
        user_id TEXT REFERENCES users(user_id) ON DELETE CASCADE,
        session_id TEXT,
        order_date TIMESTAMP,
        order_status TEXT DEFAULT 'completed',
        total_amount REAL
    );"""

    # 4. Reviews table
    create_reviews = """
    CREATE TABLE IF NOT EXISTS reviews (
        review_id TEXT PRIMARY KEY,
        product_id TEXT REFERENCES products(product_id),
        user_id TEXT REFERENCES users(user_id) ON DELETE CASCADE,
        rating REAL,
        sentiment_score REAL DEFAULT 0.0,
        review_text TEXT,
        review_date TIMESTAMP
    );"""

    # 5. Clickstream Logs
    create_clickstream = """
    CREATE TABLE IF NOT EXISTS clickstream_logs (
        event_id TEXT PRIMARY KEY,
        session_id TEXT,
        user_id TEXT,
        product_id TEXT,
        event_type TEXT,
        timestamp TIMESTAMP
    );"""

    # 6. FEATURE STORE
    create_feature_store = """
    CREATE TABLE IF NOT EXISTS product_feature_store (
        product_id TEXT PRIMARY KEY REFERENCES products(product_id),
        total_sales REAL DEFAULT 0.0,
        review_count INTEGER DEFAULT 0,
        avg_sentiment REAL DEFAULT 0.0,
        avg_rating REAL DEFAULT 0.0,
        conversion_rate REAL DEFAULT 0.0,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );"""

    # 7. Similarity Store
    create_similarity_store = """
    CREATE TABLE IF NOT EXISTS product_similarity_store (
        product_id TEXT REFERENCES products(product_id),
        similar_product_id TEXT REFERENCES products(product_id),
        similarity_score REAL,
        PRIMARY KEY (product_id, similar_product_id)
    );"""

    # 8. MLOPS LINEAGE & HEALTH LOGS (New/Updated)
    # This table bridges the gap between your Code, Data (DVC), and Model (MLflow)
    create_model_health_logs = """
    CREATE TABLE IF NOT EXISTS model_health_logs (
        id SERIAL PRIMARY KEY,
        training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        rmse REAL,
        precision REAL,
        recall REAL,
        f1_score REAL,
        model_version VARCHAR(50),
        mlflow_run_id TEXT,    -- The unique ID from MLflow
        data_hash TEXT         -- The MD5 hash from DVC
    );"""

    # Migration: Ensure existing health logs table gets the lineage columns
    migrations = [
        "ALTER TABLE model_health_logs ADD COLUMN IF NOT EXISTS mlflow_run_id TEXT;",
        "ALTER TABLE model_health_logs ADD COLUMN IF NOT EXISTS data_hash TEXT;"
    ]

    # Execution block
    try:
        with engine.begin() as conn:
            # Create all tables
            tables = [
                create_users, create_products, create_orders, 
                create_reviews, create_clickstream, create_feature_store, 
                create_similarity_store, create_model_health_logs
            ]
            for table_query in tables:
                conn.execute(text(table_query))
            
            # Apply migrations for lineage columns
            for migration_query in migrations:
                conn.execute(text(migration_query))

            logger.info("✅ Database Schema and Lineage Bridge initialized successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise

if __name__ == "__main__":
    setup_database()
