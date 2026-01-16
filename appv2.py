import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import json
import os
import sqlite3
from sqlalchemy import create_engine
from surprise import dump

# --- DATABASE & PATH CONFIGURATION ---
if os.path.exists('/.dockerenv') or os.path.exists('/opt/airflow'):
    DB_URL = "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow"
    MODEL_ROOT = "/opt/airflow/models"
    MLFLOW_DB = "/opt/airflow/mlflow_data/mlflow.db"
else:
    DB_URL = "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"
    MODEL_ROOT = os.path.expanduser("~/recomart_project/models")
    MLFLOW_DB = os.path.expanduser("~/recomart_project/mlflow_db/mlflow.db")

engine = create_engine(DB_URL)

# --- 1. DATA QUALITY & EDA HELPERS (Integrated from standalone scripts) ---
def get_data_health_metrics():
    """Logic from dq_reporter.py and eda_analysis.py integrated for live reporting."""
    try:
        df_ratings = pd.read_sql("SELECT * FROM temp_lake_ratings_raw", engine)
        if df_ratings.empty:
            return None
        
        # DQ Checks
        dq_stats = {
            "total_records": len(df_ratings),
            "missing_values": df_ratings.isnull().sum().sum(),
            "duplicates": df_ratings.duplicated().sum(),
            "range_errors": len(df_ratings[(df_ratings['rating'] < 1) | (df_ratings['rating'] > 5)]),
            "unique_users": df_ratings['user_id'].nunique(),
            "unique_items": df_ratings['product_id'].nunique()
        }
        
        # Sparsity Logic
        sparsity = (1 - (dq_stats['total_records'] / (dq_stats['unique_users'] * dq_stats['unique_items']))) * 100
        dq_stats['matrix_sparsity'] = round(sparsity, 2)
        
        return dq_stats, df_ratings
    except:
        return None, pd.DataFrame()

def get_mlflow_history():
    """Queries MLflow for historical trends and metadata tags."""
    if os.path.exists(MLFLOW_DB):
        try:
            conn = sqlite3.connect(MLFLOW_DB)
            query = """
            SELECT r.run_uuid, m.key, m.value, r.start_time, t.value as data_source
            FROM metrics m
            JOIN runs r ON m.run_uuid = r.run_uuid
            LEFT JOIN tags t ON r.run_uuid = t.run_uuid AND t.key = 'data_source'
            ORDER BY r.start_time DESC
            """
            df = pd.read_sql(query, conn)
            conn.close()
            if df.empty: return pd.DataFrame()
            df['date'] = pd.to_datetime(df['start_time'], unit='ms')
            pivot_df = df.pivot_table(index=['date', 'data_source', 'run_uuid'], columns='key', values='value').reset_index()
            return pivot_df.sort_values('date', ascending=False)
        except Exception as e:
            st.sidebar.error(f"MLflow DB Query Error: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def load_performance_metrics():
    metrics_path = os.path.join(MODEL_ROOT, "latest_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f: return json.load(f)
        except: return None
    return None

# --- DASHBOARD SETUP ---
st.set_page_config(page_title="RecoMart V3 AI Console", layout="wide", page_icon="🚀")
st.title("🚀 RecoMart V3: Production Intelligence")

# --- TOP LEVEL METRIC CARDS ---
metrics = load_performance_metrics()
k1, k2, k3, k4 = st.columns(4)
if metrics:
    k1.metric("Current Precision", f"{metrics.get('precision', 0)*100:.1f}%")
    k2.metric("Current Recall", f"{metrics.get('recall', 0)*100:.1f}%")
    k3.metric("Last Training Sync", str(metrics.get('last_run', 'N/A')).split(" ")[0])
    k4.metric("Model Status", "Healthy ✅")

st.divider()

# --- MAIN DASHBOARD TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Business EDA", "📉 Model Drift Tracker", "⚡ Conversion Funnel", "🔍 Data Health & DQ"])

with tab1:
    st.subheader("Sentiment vs. Revenue Analysis")
    try:
        df_eda = pd.read_sql("SELECT p.product_name, pfs.total_sales, pfs.avg_sentiment, pfs.review_count, p.category FROM product_feature_store pfs JOIN products p ON pfs.product_id = p.product_id", engine)
        fig = px.scatter(df_eda, x="avg_sentiment", y="total_sales", size="review_count", color="category", hover_name="product_name")
        st.plotly_chart(fig, use_container_width=True)
    except: st.info("Run the ETL pipeline to see Business EDA.")

with tab2:
    st.subheader("MLflow Performance History")
    h_df = get_mlflow_history()
    if not h_df.empty:
        m_cols = [c for c in h_df.columns if c not in ['date', 'data_source', 'run_uuid']]
        fig_drift = px.line(h_df, x='date', y=m_cols, markers=True, title="Model Accuracy Trends")
        st.plotly_chart(fig_drift, use_container_width=True)
    else: st.info("No MLflow history detected.")

with tab3:
    st.subheader("Real-time Clickstream Journey")
    try:
        df_logs = pd.read_sql("SELECT event_type, count(*) as volume FROM clickstream_logs GROUP BY event_type", engine)
        fig_f = go.Figure(go.Funnel(y=df_logs['event_type'], x=df_logs['volume']))
        st.plotly_chart(fig_f, use_container_width=True)
    except: st.info("Clickstream table not found yet.")

with tab4:
    st.subheader("Data Integrity & Distribution")
    dq_stats, df_r = get_data_health_metrics()
    if dq_stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Ratings", dq_stats['total_records'])
        c2.metric("Missing Values", dq_stats['missing_values'], delta_color="inverse")
        c3.metric("Duplicates", dq_stats['duplicates'], delta_color="inverse")
        c4.metric("Matrix Sparsity", f"{dq_stats['matrix_sparsity']}%")
        
        st.divider()
        col_dist, col_pop = st.columns(2)
        with col_dist:
            st.write("**Rating Distribution**")
            st.plotly_chart(px.histogram(df_r, x='rating', nbins=5), use_container_width=True)
        with col_pop:
            st.write("**Top 10 Product Popularity**")
            top10 = df_r['product_id'].value_counts().head(10).reset_index()
            st.plotly_chart(px.bar(top10, x='product_id', y='count'), use_container_width=True)
    else: st.info("Staged data not found. Run the Airflow pipeline.")

# --- SIDEBAR: PERSONALIZATION & LINEAGE ---
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
except: st.sidebar.info("Sync users/models to enable personalization.")

st.sidebar.divider()
st.sidebar.subheader("📜 Model Lineage (MLflow)")
h_df = get_mlflow_history()
if not h_df.empty:
    latest = h_df.iloc[0]
    with st.sidebar.expander("Latest Version Info", expanded=True):
        st.write(f"**Source:** `{latest.get('data_source', 'N/A')}`")
        st.write(f"**RMSE:** `{latest.get('rmse', 0):.4f}`")
        st.caption(f"Run ID: {latest.get('run_uuid', 'N/A')[:8]}")
