import os
import glob
import pandas as pd
from sqlalchemy import create_engine, text
import logging

# --- CONFIG ---
DB_URL = "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"
LAKE_PATH = os.path.expanduser("~/recomart_project/data_lake/kafka/product_review")
engine = create_engine(DB_URL)

def verify_pipeline():
    print("\n" + "🔍" * 30)
    print("      RECOMART PIPELINE DIAGNOSTIC")
    print("🔍" * 30 + "\n")

    # STEP 1: Check Physical Data Lake (The 'Files' layer)
    print("📂 [1/3] CHECKING DATA LAKE FILES...")
    files = glob.glob(f"{LAKE_PATH}/**/*.jsonl", recursive=True)
    if not files:
        print("❌ FAIL: No .jsonl files found in Lake. Ingester is not saving data or path is wrong.")
        print(f"   Checked path: {LAKE_PATH}")
    else:
        print(f"✅ PASS: Found {len(files)} event files in the lake.")

    # STEP 2: Check Database Connectivity & Seeding
    print("\n🗄️  [2/3] CHECKING DATABASE TABLES...")
    try:
        with engine.connect() as conn:
            users = conn.execute(text("SELECT COUNT(*) FROM users")).scalar()
            products = conn.execute(text("SELECT COUNT(*) FROM products")).scalar()
            
            # Check if temp table exists (the one the DAG creates)
            table_check = conn.execute(text(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'temp_lake_ratings_raw')"
            )).scalar()
            
            print(f"✅ Users: {users} | Products: {products}")
            if table_check:
                rows = conn.execute(text("SELECT COUNT(*) FROM temp_lake_ratings_raw")).scalar()
                print(f"✅ Staging Table: EXISTS ({rows} rows)")
            else:
                print("⚠️  Staging Table: MISSING (The DAG 'process_sentiment_scores' hasn't run yet)")
    except Exception as e:
        print(f"❌ FAIL: Database connection error: {e}")

    # STEP 3: Check SVD Readiness (The 'ML' layer)
    print("\n🧠 [3/3] CHECKING ML TRAINING READINESS...")
    try:
        df = pd.read_sql("SELECT user_id, product_id, rating FROM temp_lake_ratings_raw", engine)
        unique_users = df['user_id'].nunique()
        unique_prods = df['product_id'].nunique()
        
        print(f"📊 Matrix Density: {len(df)} interactions | {unique_users} users | {unique_prods} products")
        
        if len(df) < 5:
            print("❌ FAIL: Not enough data for SVD training. Need at least 5-10 interactions.")
        else:
            print("✅ PASS: Data density is sufficient for Collaborative Filtering.")
            
    except Exception:
        print("❌ FAIL: Cannot calculate matrix density. (Staging table likely empty)")

    print("\n" + "="*60)
    print("💡 NEXT STEP: If all [PASS], go to Airflow UI and 'Clear' the failed task.")
    print("="*60 + "\n")

if __name__ == "__main__":
    verify_pipeline()
