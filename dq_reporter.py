import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"
engine = create_engine(DB_URL)

def generate_quality_report():
    print("🔍 RECOMART DATA QUALITY REPORT")
    print("="*40)
    
    checks = {
        "Users": "SELECT * FROM users",
        "Products": "SELECT * FROM products",
        "Ratings": "SELECT * FROM temp_lake_ratings_raw"
    }
    
    for name, query in checks.items():
        df = pd.read_sql(query, engine)
        
        print(f"\n📁 Table: {name}")
        print(f"   - Total Records: {len(df)}")
        
        # 1. Missing Values
        nulls = df.isnull().sum().sum()
        print(f"   - Missing Values: {nulls} {'✅' if nulls == 0 else '❌'}")
        
        # 2. Duplicates
        dupes = df.duplicated().sum()
        print(f"   - Duplicate Rows: {dupes} {'✅' if dupes == 0 else '⚠️'}")
        
        # 3. Range Checks (Specific to Ratings)
        if name == "Ratings":
            out_of_range = df[(df['rating'] < 1) | (df['rating'] > 5)]
            print(f"   - Rating Range (1-5) Check: {len(out_of_range)} failures {'✅' if len(out_of_range) == 0 else '❌'}")

    # 4. Schema Validation (Checking if IDs are unique)
    with engine.connect() as conn:
        res = conn.execute(text("SELECT COUNT(product_id) FROM products")).scalar()
        dist = conn.execute(text("SELECT COUNT(DISTINCT product_id) FROM products")).scalar()
        print(f"\n🆔 Primary Key Integrity (Products): {'✅ PASS' if res == dist else '❌ FAIL'}")

if __name__ == "__main__":
    generate_quality_report()
