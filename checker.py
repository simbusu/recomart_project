import pandas as pd
from sqlalchemy import create_engine, text
import logging
from datetime import datetime

# --- CONFIGURATION ---
DB_URL = "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"
engine = create_engine(DB_URL)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def run_integrity_check():
    print("\n" + "="*95)
    print(f"🚀 RECOMART V3 (AI-READY) SYSTEM INTEGRITY CHECK | {datetime.now().strftime('%H:%M:%S')}")
    print("="*95)

    with engine.connect() as conn:
        # 1. Database & Quality Stats
        user_count = conn.execute(text("SELECT COUNT(*) FROM users")).scalar()
        logger.info(f"📊 [DATABASE] Users: {user_count} | Cities: {conn.execute(text('SELECT COUNT(DISTINCT city) FROM users')).scalar()}")
        
        neg_rev = conn.execute(text("SELECT COUNT(*) FROM product_feature_store WHERE consolidated_sales < 0")).scalar()
        logger.info(f"🛡️  [QUALITY] Negative Revenue records blocked: {neg_rev}\n")

        # 2. Advanced Reputation View (Original vs Updated)
        print("🌟 [REPUTATION SCORE] API Baseline vs. Community Feedback:")
        # We compare p.rating (API) with pfs.avg_rating (Batch Lake)
        reputation_query = text("""
            SELECT 
                p.product_name,
                p.rating as original_rating,
                ROUND(pfs.avg_rating::numeric, 2) as updated_rating,
                ROUND(pfs.sentiment_score::numeric, 2) as sentiment,
                ROUND(
                    ((pfs.avg_rating / 5.0) * 0.6 + ((pfs.sentiment_score + 1.0) / 2.0) * 0.4)::numeric, 
                    3
                ) as reputation_index
            FROM product_feature_store pfs
            JOIN products p ON pfs.product_id = p.product_id
            WHERE pfs.avg_rating > 0
            ORDER BY reputation_index DESC
            LIMIT 8
        """)
        
        df_rep = pd.read_sql(reputation_query, conn)
        if not df_rep.empty:
            # Rename for clarity in output
            df_rep.columns = ['Product', 'Orig ★', 'Updated ★', 'Sentiment', 'Rep Score']
            print(df_rep.to_string(index=False))
        else:
            logger.warning("     ⚠️ No feedback data found. Ensure the Airflow DAG has run successfully.")

        # 3. Funnel & Revenue (Condensed)
        print("\n📈 [FUNNEL] Top Conversion:")
        funnel_query = text("""
            SELECT p.product_name, ROUND((pfs.conversion_rate * 100)::numeric, 2) || '%' as conversion
            FROM product_feature_store pfs
            JOIN products p ON pfs.product_id = p.product_id
            ORDER BY pfs.conversion_rate DESC LIMIT 3
        """)
        print(pd.read_sql(funnel_query, conn).to_string(index=False))

        print("\n💰 [SPEED LAYER] Top Revenue:")
        rev_query = text("""
            SELECT p.product_name, '$' || ROUND(pfs.consolidated_sales::numeric, 2) as revenue
            FROM product_feature_store pfs
            JOIN products p ON pfs.product_id = p.product_id
            ORDER BY pfs.consolidated_sales DESC LIMIT 3
        """)
        print(pd.read_sql(rev_query, conn).to_string(index=False))

    print("\n" + "="*95)
    print("💡 Reputation = 60% Community Rating + 40% AI Sentiment Analysis")
    print("="*95 + "\n")

if __name__ == "__main__":
    run_integrity_check()
