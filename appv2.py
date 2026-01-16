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

# --- 1. DATA HELPERS ---
def get_data_health_metrics():
    try:
        df_ratings = pd.read_sql("SELECT * FROM temp_lake_ratings_raw", engine)
        if df_ratings.empty: return None, pd.DataFrame()
        dq_stats = {
            "total_records": len(df_ratings),
            "missing_values": df_ratings.isnull().sum().sum(),
            "duplicates": df_ratings.duplicated().sum(),
            "unique_users": df_ratings['user_id'].nunique(),
            "unique_items": df_ratings['product_id'].nunique()
        }
        sparsity = (1 - (dq_stats['total_records'] / (dq_stats['unique_users'] * dq_stats['unique_items']))) * 100
        dq_stats['matrix_sparsity'] = round(sparsity, 2)
        return dq_stats, df_ratings
    except: return None, pd.DataFrame()

def get_mlflow_history():
    if os.path.exists(MLFLOW_DB):
        try:
            conn = sqlite3.connect(MLFLOW_DB)
            query = """
            SELECT r.run_uuid, m.key, m.value, r.start_time, 
                   t1.value as data_source, t2.value as dvc_hash
            FROM metrics m
            JOIN runs r ON m.run_uuid = r.run_uuid
            LEFT JOIN tags t1 ON r.run_uuid = t1.run_uuid AND t1.key = 'data_source'
            LEFT JOIN tags t2 ON r.run_uuid = t2.run_uuid AND t2.key = 'dvc_hash'
            WHERE r.status = 'FINISHED'
            ORDER BY r.start_time ASC
            """
            df = pd.read_sql(query, conn)
            conn.close()
            if df.empty: return pd.DataFrame()
            df['date'] = pd.to_datetime(df['start_time'], unit='ms')
            df['key'] = df['key'].str.lower().str.strip()
            return df.pivot_table(index=['date', 'data_source', 'dvc_hash', 'run_uuid'], 
                                  columns='key', values='value').reset_index()
        except: return pd.DataFrame()
    return pd.DataFrame()

def load_performance_metrics():
    metrics_path = os.path.join(MODEL_ROOT, "latest_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f: return json.load(f)
        except: return None
    return None

# --- DASHBOARD SETUP ---
st.set_page_config(page_title="RecoMart V3 Console", layout="wide", page_icon="🚀")
st.title("🚀 RecoMart V3: Production Intelligence")

# --- TOP LEVEL METRIC CARDS ---
metrics = load_performance_metrics()
h_df = get_mlflow_history()
k1, k2, k3, k4 = st.columns(4)

if metrics:
    k1.metric("Current Precision", f"{metrics.get('precision', 0)*100:.1f}%")
    k2.metric("Current Recall", f"{metrics.get('recall', 0)*100:.1f}%")
    k3.metric("Last Training Sync", str(metrics.get('last_run', 'N/A')).split(" ")[0])
    k4.metric("Model Status", "Healthy ✅" if metrics.get('rmse', 2) < 1.45 else "Drift ⚠️")

st.divider()

# --- SIDEBAR: PERSONALIZATION & LINEAGE ---
st.sidebar.header("🎯 AI Personalization")
try:
    user_list = pd.read_sql("SELECT DISTINCT user_id FROM users LIMIT 50", engine)['user_id'].tolist()
    sel_user = st.sidebar.selectbox("Select User ID", user_list)
    if st.sidebar.button("Generate Picks"):
        m_path = os.path.join(MODEL_ROOT, "svd_v1.pkl")
        if os.path.exists(m_path):
            _, algo = dump.load(m_path)
            prods = pd.read_sql("""
                SELECT p.product_id, p.product_name, f.avg_sentiment 
                FROM products p 
                LEFT JOIN product_feature_store f ON p.product_id = f.product_id
            """, engine)
            prods['score'] = prods.apply(lambda x: (algo.predict(str(sel_user), str(x['product_id'])).est * 0.7) + ((x['avg_sentiment'] or 0) * 0.3), axis=1)
            st.sidebar.write(f"**Top Picks for {sel_user}:**")
            st.sidebar.table(prods.sort_values('score', ascending=False).head(5)[['product_name', 'score']])
        else:
            st.sidebar.error("Model file `svd_v1.pkl` not found.")
except:
    st.sidebar.info("Sync users to enable personalization.")

st.sidebar.divider()
st.sidebar.header("🛡️ Model Lineage")
if not h_df.empty:
    latest = h_df.iloc[-1]
    st.sidebar.success(f"**DVC Hash:**\n`{latest.get('dvc_hash', 'N/A')[:12]}`")
    st.sidebar.caption(f"**Source:** {latest.get('data_source', 'N/A')}")

# --- MAIN DASHBOARD TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Business EDA", "📉 Model Drift", "⚡ Funnel", "🔍 Data Health", "📜 Audit Log"])

with tab1:
    st.subheader("🏙️ Product Sentiment vs. Revenue Analysis")
    try:
        df_eda = pd.read_sql("""
            SELECT p.product_name, pfs.total_sales, pfs.avg_sentiment, pfs.review_count, p.category 
            FROM product_feature_store pfs 
            JOIN products p ON pfs.product_id = p.product_id
        """, engine)
        
        if not df_eda.empty:
            fig = px.scatter(
                df_eda, 
                x="avg_sentiment", 
                y="total_sales", 
                size="review_count", 
                color="category", 
                hover_name="product_name",
                title="Revenue Drivers: Sentiment & Sales",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data in Feature Store.")
    except:
        st.info("Run the ETL pipeline to sync the Feature Store for Sentiment analysis.")

with tab2:
    st.subheader("📉 Model Performance Tracking")
    if not h_df.empty:
        fig_drift = px.line(h_df, x='date', y='rmse', markers=True, title="RMSE Trend", template="plotly_dark")
        fig_drift.add_hline(y=1.45, line_dash="dot", line_color="red", annotation_text="Drift Limit")
        st.plotly_chart(fig_drift, use_container_width=True)
    else: st.info("No MLflow history.")

with tab3:
    st.subheader("⚡ Conversion Funnel")
    try:
        df_logs = pd.read_sql("SELECT event_type, count(*) as volume FROM clickstream_logs GROUP BY event_type", engine)
        fig_f = go.Figure(go.Funnel(y=df_logs['event_type'], x=df_logs['volume']))
        st.plotly_chart(fig_f, use_container_width=True)
    except: st.info("Clickstream data missing.")

with tab4:
    st.subheader("🔍 Data Health")
    dq, df_r = get_data_health_metrics()
    if dq:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", dq['total_records'])
        c2.metric("Matrix Sparsity", f"{dq['matrix_sparsity']}%")
        c3.metric("Duplicates", dq['duplicates'])
        st.plotly_chart(px.histogram(df_r, x='rating', title="Rating Distribution"), use_container_width=True)

with tab5:
    st.subheader("📜 Audit Log")
    if not h_df.empty:
        history = h_df.sort_values('date', ascending=False).head(5).copy()
        history['RMSE'] = history['rmse'].map('{:,.4f}'.format)
        st.table(history[['date', 'dvc_hash', 'run_uuid', 'RMSE']])
