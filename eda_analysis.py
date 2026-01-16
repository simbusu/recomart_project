import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import os

# --- CONNECTION ---
DB_URL = "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"
engine = create_engine(DB_URL)

def run_eda():
    print("📊 Starting Exploratory Data Analysis...")
    
    # Load Data
    df_ratings = pd.read_sql("SELECT user_id, product_id, rating FROM temp_lake_ratings_raw", engine)
    df_products = pd.read_sql("SELECT product_id, category FROM products", engine)
    
    if df_ratings.empty:
        print("⚠️ No data found in temp_lake_ratings_raw. Run the pipeline first.")
        return

    # 1. Interaction Distribution (Rating Spread)
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_ratings, x='rating', palette='viridis')
    plt.title('Distribution of User Ratings')
    plt.savefig('eda_rating_dist.png')
    
    # 2. Item Popularity (Top 10 Products)
    plt.figure(figsize=(10, 6))
    top_items = df_ratings['product_id'].value_counts().head(10)
    top_items.plot(kind='bar', color='skyblue')
    plt.title('Top 10 Most Interacted Products')
    plt.ylabel('Interaction Count')
    plt.savefig('eda_item_popularity.png')

    # 3. Sparsity Pattern Calculation
    n_users = df_ratings['user_id'].nunique()
    n_items = df_ratings['product_id'].nunique()
    n_ratings = len(df_ratings)
    
    possible_interactions = n_users * n_items
    sparsity = (1 - (n_ratings / possible_interactions)) * 100
    
    print(f"📈 Sparsity Analysis:")
    print(f"   - Unique Users: {n_users}")
    print(f"   - Unique Items: {n_items}")
    print(f"   - Total Interactions: {n_ratings}")
    print(f"   - Matrix Sparsity: {sparsity:.2f}%")

    # 4. Interaction Heatmap (Sample)
    sample_users = df_ratings['user_id'].unique()[:20]
    sample_df = df_ratings[df_ratings['user_id'].isin(sample_users)]
    pivot = sample_df.pivot_table(index='user_id', columns='product_id', values='rating')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, cmap='YlGnBu', cbar_kws={'label': 'Rating'})
    plt.title('User-Item Interaction Heatmap (Sample)')
    plt.savefig('eda_sparsity_heatmap.png')
    
    print("✅ EDA complete. Plots saved as PNG files.")

if __name__ == "__main__":
    run_eda()
