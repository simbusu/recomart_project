import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from fpdf import FPDF
from datetime import datetime

# --- CONFIGURATION ---
DB_URL = "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"
engine = create_engine(DB_URL)

class FullEDAReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Recomart: 360-Degree Data Insights Report', 0, 1, 'C')
        self.set_font('Arial', 'I', 9)
        self.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'C')
        self.ln(5)

    def add_section(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 10, f" {title}", 0, 1, 'L', fill=True)
        self.ln(4)

def run_comprehensive_eda():
    # Load all relevant data
    df_r = pd.read_sql("SELECT * FROM temp_lake_ratings_raw", engine)
    df_p = pd.read_sql("SELECT * FROM products", engine)
    df_u = pd.read_sql("SELECT * FROM users", engine)
    
    pdf = FullEDAReport()
    pdf.add_page()

    # --- 1. USER & TRANSACTION ANALYSIS ---
    pdf.add_section("1. User Engagement & Transaction Volume")
    user_activity = df_r.groupby('user_id').size()
    
    plt.figure(figsize=(8, 4))
    sns.histplot(user_activity, bins=30, kde=True, color='teal')
    plt.title('Distribution of Interactions per User')
    plt.savefig('eda_user_activity.png')
    pdf.image('eda_user_activity.png', x=15, w=170)
    
    avg_act = user_activity.mean()
    pdf.set_font('Arial', '', 10)
    pdf.ln(5)
    pdf.multi_cell(0, 6, f"Summary:\n- Average interactions per user: {avg_act:.2f}\n"
                         f"- Total Transactions: {len(df_r)}\n"
                         f"- Active User Base: {len(user_activity)}")

    # --- 2. PRODUCT & LONG TAIL ANALYSIS ---
    pdf.add_page()
    pdf.add_section("2. Product Catalog & Popularity (Long Tail)")
    prod_counts = df_r['product_id'].value_counts()
    cum_vol = np.cumsum(prod_counts.values) / np.sum(prod_counts.values)
    
    plt.figure(figsize=(8, 4))
    plt.plot(cum_vol, color='orange', linewidth=2)
    plt.axhline(y=0.8, color='r', linestyle='--')
    plt.title('Long Tail: Cumulative Interaction Volume')
    plt.savefig('eda_long_tail.png')
    pdf.image('eda_long_tail.png', x=15, w=170)
    
    # --- 3. MATRIX SPARSITY (The Connectivity) ---
    pdf.ln(10)
    pdf.add_section("3. Matrix Connectivity (Sparsity)")
    n_users = df_r['user_id'].nunique()
    n_prods = df_r['product_id'].nunique()
    sparsity = (1 - (len(df_r) / (n_users * n_prods))) * 100
    
    # Interaction Heatmap (Sample)
    sample_users = df_r['user_id'].unique()[:20]
    sample_df = df_r[df_r['user_id'].isin(sample_users)]
    pivot = sample_df.pivot_table(index='user_id', columns='product_id', values='rating')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, cmap='YlGnBu', cbar=False)
    plt.title('User-Item Sparsity Heatmap (Sample)')
    plt.savefig('eda_sparsity_map.png')
    pdf.image('eda_sparsity_map.png', x=30, w=140)
    pdf.ln(5)
    pdf.cell(0, 10, f"Calculated Matrix Sparsity: {sparsity:.2f}%", 0, 1)

    # --- 4. REVIEW & SENTIMENT ANALYSIS ---
    pdf.add_page()
    pdf.add_section("4. Review Quality & Sentiment Correlation")
    
    plt.figure(figsize=(8, 4))
    sns.regplot(data=df_r, x='sentiment', y='rating', scatter_kws={'alpha':0.2}, line_kws={'color':'red'})
    plt.title('Review Sentiment vs. Explicit Star Rating')
    plt.savefig('eda_sentiment.png')
    pdf.image('eda_sentiment.png', x=15, w=170)
    
    corr = df_r['sentiment'].corr(df_r['rating'])
    pdf.ln(5)
    pdf.cell(0, 10, f"Correlation Coefficient (Sentiment/Rating): {corr:.2f}", 0, 1)

    # Save final PDF
    output_name = f"recomart_full_eda_{datetime.now().strftime('%Y%m%d')}.pdf"
    pdf.output(output_name)
    print(f"✅ Full EDA Report Generated: {output_name}")

if __name__ == "__main__":
    run_comprehensive_eda()
