import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sqlite3
import subprocess
from sqlalchemy import create_engine
from surprise import dump

# --- 1. CONFIGURATION & DATABASE CONNECTION ---
if os.path.exists('/.dockerenv') or os.path.exists('/opt/airflow'):
    DB_URL = "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow"
    MODEL_ROOT = "/opt/airflow/models"
    MLFLOW_URL = "http://localhost:5000" # URL for clicking through to MLflow
else:
    DB_URL = "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"
    MODEL_ROOT = os.path.expanduser("~/recomart_project/models")
    MLFLOW_URL = "http://localhost:5000"

engine = create_engine(DB_URL)

# --- 2. DATA RETRIEVAL FUNCTIONS ---
def get_live_inventory_stats():
    try:
        n_users = pd.read_sql("SELECT COUNT(*) FROM users", engine).iloc[0,0]
        n_prods = pd.read_sql("SELECT COUNT(*) FROM products", engine).iloc[0,0]
        n_reviews = pd.read_sql("SELECT COUNT(*) FROM reviews", engine).iloc[0,0]
        return n_users, n_prods, n_reviews
    except: return 0, 0, 0

def get_detailed_metrics():
    """Pulls Metrics + MLflow Run ID + DVC Data Hash."""
    try:
        # Added mlflow_run_id and data_hash to the selection
        query = """
            SELECT training_date as date, rmse, precision, recall, f1_score, 
                   mlflow_run_id, data_hash, model_version
            FROM model_health_logs
            ORDER BY training_date ASC
        """
        return pd.read_sql(query, engine)
    except: return pd.DataFrame()

def get_deep_analytics_data():
    try:
        query = """
            SELECT r.user_id, r.product_id, r.rating, r.review_date,
                   COALESCE(u.city, 'Unknown') as city,
                   p.category, p.product_name
            FROM reviews r
            LEFT JOIN users u ON r.user_id = u.user_id
            JOIN products p ON r.product_id = p.product_id
        """
        df = pd.read_sql(query, engine)
        df['review_date'] = pd.to_datetime(df['review_date'])
        return df
    except: return pd.DataFrame()

def get_feature_store_data():
    try:
        return pd.read_sql("""
            SELECT p.product_name, pfs.total_sales, pfs.avg_sentiment, pfs.review_count, p.category
            FROM product_feature_store pfs
            JOIN products p ON pfs.product_id = p.product_id
        """, engine)
    except: return pd.DataFrame()

# --- 3. DASHBOARD UI ---
st.set_page_config(page_title="RecoMart V3 MLOps Console", layout="wide", page_icon="🚀")
st.title("🚀 RecoMart V3: MLOps & Lineage Intelligence")

# --- TOP LEVEL METRIC CARDS ---
n_u, n_p, n_r = get_live_inventory_stats()
sparsity = round((1 - (n_r / (n_u * n_p if n_u*n_p > 0 else 1))) * 100, 2)
metrics_df = get_detailed_metrics()

# Main Row Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Catalog Users", f"{n_u:,}")
m2.metric("Total Products", f"{n_p:,}")
m3.metric("Live Interactions", f"{n_r:,}")
m4.metric("Matrix Sparsity", f"{sparsity}%")

# Sub-Row Metrics (Precision & Recall)
if not metrics_df.empty:
    latest = metrics_df.iloc[-1]
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Model Precision", f"{round(latest['precision']*100, 2)}%")
    s2.metric("Model Recall", f"{round(latest['recall']*100, 2)}%")
    s1.caption(f"Data Hash: `{latest['data_hash']}`") # Lineage Indicator
    s2.caption(f"Run ID: `{latest['mlflow_run_id'][:8] if latest['mlflow_run_id'] else 'N/A'}`")

st.divider()

# --- SIDEBAR (AI Personalization) ---
st.sidebar.header("🎯 AI Personalization")
try:
    user_list = pd.read_sql("SELECT DISTINCT user_id FROM users LIMIT 50", engine)['user_id'].tolist()
    sel_user = st.sidebar.selectbox("Select User ID", user_list)
    if st.sidebar.button("Generate Picks"):
        m_path = os.path.join(MODEL_ROOT, "svd_v1.pkl")
        if os.path.exists(m_path):
            _, algo = dump.load(m_path)
            prods = pd.read_sql("SELECT p.product_id, p.product_name, f.avg_sentiment FROM products p LEFT JOIN product_feature_store f ON p.product_id = f.product_id", engine)
            prods['score'] = prods.apply(lambda x: (algo.predict(str(sel_user), str(x['product_id'])).est * 0.7) + ((x['avg_sentiment'] or 0) * 0.3), axis=1)
            st.sidebar.table(prods.sort_values('score', ascending=False).head(5)[['product_name', 'score']])
except: st.sidebar.info("Sync users to enable AI picks.")

# --- 4. MAIN DASHBOARD TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Business Intelligence", "🔬 Deep Analytics", "📉 Model Drift",
    "⚡ Live Funnel", "🔍 Data Health", "📜 MLOps Audit"
])

# (Tabs 1-5 remain identical to your original code for stability)
with tab1:
    st.subheader("🏙️ Strategic Sentiment & Revenue Analysis")
    df_eda = get_feature_store_data()
    if not df_eda.empty:
        st.plotly_chart(px.scatter(df_eda, x="avg_sentiment", y="total_sales", size="review_count", color="category",
                                   hover_name="product_name", template="plotly_dark"), use_container_width=True)

with tab2:
    st.subheader("🔬 Region & Category Drill-down")
    df_adv = get_deep_analytics_data()
    if not df_adv.empty:
        heatmap_data = df_adv.groupby(['city', 'category']).size().reset_index(name='interactions')
        st.plotly_chart(px.density_heatmap(heatmap_data, x="city", y="category", z="interactions", color_continuous_scale="Viridis", template="plotly_dark"), use_container_width=True)

with tab3:
    st.subheader("📉 Performance & Quality Drift")
    if not metrics_df.empty:
        st.plotly_chart(px.line(metrics_df, x='date', y=['precision', 'recall'], markers=True, template="plotly_dark"), use_container_width=True)
        fig_drift = px.area(metrics_df, x='date', y='rmse', template="plotly_dark")
        fig_drift.add_hline(y=1.45, line_dash="dot", line_color="red")
        st.plotly_chart(fig_drift, use_container_width=True)

with tab4:
    st.subheader("⚡ Live Clickstream Funnel")
    try:
        df_funnel = pd.read_sql("SELECT event_type, count(*) as volume FROM clickstream_logs GROUP BY event_type", engine)
        st.plotly_chart(go.Figure(go.Funnel(y=df_funnel['event_type'], x=df_funnel['volume'])), use_container_width=True)
    except: st.info("No Clickstream data found.")

with tab5:
    st.subheader("🔍 Data Health & Sparsity")
    df_health = pd.read_sql("SELECT rating FROM reviews", engine)
    if not df_health.empty:
        st.plotly_chart(px.histogram(df_health, x='rating', template="plotly_dark", nbins=5), use_container_width=True)

# --- UPDATED TAB 6: MLOps Audit with Lineage ---
with tab6:
    st.subheader("📜 MLOps Lineage Audit Log")
    if not metrics_df.empty:
        # Create a display-friendly dataframe
        audit_df = metrics_df.copy().sort_values('date', ascending=False)
        
        # Add a "View in MLflow" Link column
        audit_df['mlflow_link'] = audit_df['mlflow_run_id'].apply(
            lambda x: f"{MLFLOW_URL}/#/experiments/1/runs/{x}" if x else "N/A"
        )
        
        st.write("#### 🧪 Model Versioning & Data Provenance")
        st.dataframe(
            audit_df[['date', 'model_version', 'rmse', 'precision', 'recall', 'data_hash', 'mlflow_run_id']],
            use_container_width=True
        )
        
        st.info("💡 The **Data Hash** represents the DVC snapshot used for training. The **Run ID** links to the full MLflow artifact registry.")
    else:
        st.info("No training metadata available yet.")
