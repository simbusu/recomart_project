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
    Ensures all tables for ingestion, feature storage, and ML similarity are present.
    """
    # Environment-based connection string
    if os.path.exists('/.dockerenv') or os.path.exists('/opt/airflow'):
        db_url = "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow"
    else:
        db_url = "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"

    logger.info(f"Connecting to database: {db_url}")
    engine = create_engine(db_url)

    # 1. Users table for demographic and signup data
    create_users = """
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        name TEXT,
        email TEXT,
        gender TEXT,
        city TEXT,
        signup_date DATE
    );"""

    # 2. Products table for base catalog information
    create_products = """
    CREATE TABLE IF NOT EXISTS products (
        product_id TEXT PRIMARY KEY,
        product_name TEXT,
        category TEXT,
        price REAL,
        rating REAL
    );"""

    # 3. Orders table including session and transaction totals
    create_orders = """
    CREATE TABLE IF NOT EXISTS orders (
        order_id TEXT PRIMARY KEY,
        user_id TEXT REFERENCES users(user_id) ON DELETE CASCADE,
        session_id TEXT,
        order_date TIMESTAMP,
        order_status TEXT DEFAULT 'completed',
        total_amount REAL
    );"""

    # 4. Reviews table with sentiment tracking
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

    # 5. Clickstream Logs for user behavior tracking
    create_clickstream = """
    CREATE TABLE IF NOT EXISTS clickstream_logs (
        event_id TEXT PRIMARY KEY,
        session_id TEXT,
        user_id TEXT,
        product_id TEXT,
        event_type TEXT,
        timestamp TIMESTAMP
    );"""

    # 6. FEATURE STORE: Updated to support daily DAG and Dashboard requirements
    # Aligned with transform_to_feature_store in batch_etl_dag.py
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

    # 7. Similarity Store for content-based recommendation matrices
    create_similarity_store = """
    CREATE TABLE IF NOT EXISTS product_similarity_store (
        product_id TEXT REFERENCES products(product_id),
        similar_product_id TEXT REFERENCES products(product_id),
        similarity_score REAL,
        PRIMARY KEY (product_id, similar_product_id)
    );"""

    # Execution block with transaction management
    try:
        with engine.begin() as conn:
            conn.execute(text(create_users))
            conn.execute(text(create_products))
            conn.execute(text(create_orders))
            conn.execute(text(create_reviews))
            conn.execute(text(create_clickstream))
            conn.execute(text(create_feature_store))
            conn.execute(text(create_similarity_store))
            
            logger.info("✅ Database Schema initialized and unified for ML production.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise

if __name__ == "__main__":
    setup_database()
