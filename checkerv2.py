import os
import glob
import pandas as pd
from sqlalchemy import create_engine, text
import logging
from datetime import datetime

# --- CONFIGURATION ---
DB_URL = "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"
LAKE_PATH = os.path.expanduser("~/recomart_project/data_lake/kafka/product_review")
MODEL_PATH = os.path.expanduser("~/recomart_project/models/svd_v1.pkl")
engine = create_engine(DB_URL)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def run_checks():
    print("\n" + "="*95)
    print(f"🚀 RECOMART V3 (AI-READY) SYSTEM INTEGRITY CHECK | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*95)

    # --- 1. PHYSICAL PIPELINE CHECK ---
    print("\n📂 [PIPELINE DIAGNOSTICS]")
    
    # Check Lake
    lake_files = glob.glob(f"{LAKE_PATH}/**/*.jsonl", recursive=True)
    lake_status = "✅ PASS" if lake_files else "❌ FAIL"
    print(f"{lake_status}: Data Lake contains {len(lake_files)} event files.")

    # Check Model
    model_exists = os.path.exists(MODEL_PATH)
    model_status = "✅ PASS" if model_exists else "⚠️  WAITING"
    print(f"{model_status}: ML Model artifact found at {MODEL_PATH}" if model_exists else f"{model_status}: SVD model not found yet.")

    # --- 2. DATABASE & QUALITY STATS ---
    print("\n📊 [DATABASE & QUALITY]")
    try:
        with engine.connect() as conn:
            user_count = conn.execute(text("SELECT COUNT(*) FROM users")).scalar()
            prod_count = conn.execute(text("SELECT COUNT(*) FROM products")).scalar()
            logger.info(f"Users: {user_count} | Products: {prod_count}")

            neg_rev = conn.execute(text("SELECT COUNT(*) FROM product_feature_store WHERE total_sales < 0")).scalar()
            logger.info(f"Negative Revenue records blocked: {neg_rev}")

            # --- 3. REPUTATION VIEW ---
            print("\n🌟 [REPUTATION SCORE] API Baseline vs. Community Feedback:")
            reputation_query = text("""
                SELECT
                    p.product_name,
                    p.rating as original_rating,
                    ROUND(pfs.avg_sentiment::numeric, 2) as sentiment,
                    pfs.review_count,
                    ROUND(
                        ((p.rating / 5.0) * 0.4 + ((pfs.avg_sentiment + 1.0) / 2.0) * 0.6)::numeric,
                        3
                    ) as reputation_index
                FROM product_feature_store pfs
                JOIN products p ON pfs.product_id = p.product_id
                WHERE pfs.review_count > 0
                ORDER BY reputation_index DESC
                LIMIT 8
            """)

            df_rep = pd.read_sql(reputation_query, conn)
            if not df_rep.empty:
                df_rep.columns = ['Product', 'Orig ★', 'Sentiment', 'Reviews', 'Rep Score']
                print(df_rep.to_string(index=False))
            else:
                print("⚠️  No combined feedback data found in feature store.")

            # --- 4. TOP REVENUE ---
            print("\n💰 [SPEED LAYER] Top Revenue (Real-time):")
            rev_query = text("""
                SELECT p.product_name, '$' || ROUND(pfs.total_sales::numeric, 2) as revenue
                FROM product_feature_store pfs
                JOIN products p ON pfs.product_id = p.product_id
                ORDER BY pfs.total_sales DESC LIMIT 3
            """)
            print(pd.read_sql(rev_query, conn).to_string(index=False))

    except Exception as e:
        print(f"❌ DATABASE ERROR: {e}")

    print("\n" + "="*95)
    print("💡 Formula: Reputation = 40% API Rating + 60% AI Sentiment Analysis (Batch Processed)")
    print("="*95 + "\n")

if __name__ == "__main__":
    run_checks()
