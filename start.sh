#!/bin/bash

# --- 1. CONFIGURATION ---
export AIRFLOW_HOME=~/recomart_project
export VENV_BIN=$AIRFLOW_HOME/venv/bin

# Airflow Connection Strings
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@localhost:5432/airflow
export AIRFLOW__CORE__EXECUTOR=LocalExecutor
export AIRFLOW__CORE__LOAD_EXAMPLES=False

# MLflow Path Fixes
export MLFLOW_SET_DESTINATION=/home/ubuntu/recomart_project/mlruns
export MLFLOW_TRACKING_URI=http://172.17.0.1:5000

echo "🚀 Starting RecoMart V3 Infrastructure..."

# --- 2. ENVIRONMENT REFRESH ---
cd $AIRFLOW_HOME
source venv/bin/activate

echo -n "📦 Synchronizing Python dependencies... "
pip install "numpy<2.0.0" pandas sqlalchemy nltk mlflow==2.15.1 dvc pyyaml scikit-surprise psycopg2-binary streamlit plotly -q
echo "✅ Done."

# --- 3. INFRASTRUCTURE & DATA SYNC ---
echo -n "🔗 Syncing Data (DVC)... "
dvc pull -q > /dev/null 2>&1
echo "✅ Ready."

echo -n "🐳 Bringing up Docker Services (DB/Kafka/MLflow)... "
docker compose up -d > /dev/null 2>&1
until pg_isready -h localhost -p 5432 -U airflow > /dev/null 2>&1; do
    printf "."
    sleep 1
done
echo " ✅ Ready."

# --- 4. AIRFLOW PREP & START ---
echo -n "⚙️ Migrating DB... "
$VENV_BIN/airflow db migrate > /dev/null 2>&1
$VENV_BIN/python3 init_db.py > /dev/null 2>&1
echo "✅ Done."

echo "🌪️ Starting Airflow Standalone..."
pkill -f "airflow" > /dev/null 2>&1
rm -f $AIRFLOW_HOME/airflow-webserver.pid
nohup $VENV_BIN/airflow standalone > airflow_standalone.log 2>&1 &

echo -n "⏳ Waiting for Pipeline Trigger... "
sleep 15
$VENV_BIN/airflow dags unpause recomart_full_pipeline > /dev/null 2>&1
$VENV_BIN/airflow dags trigger recomart_full_pipeline > /dev/null 2>&1
echo "✅ Done."

# --- 5. START DASHBOARD ---
echo "📈 Launching Production Dashboard..."
pkill -f "streamlit" > /dev/null 2>&1
nohup streamlit run appv2.py --server.port 8501 > streamlit.log 2>&1 &

echo "------------------------------------------------------"
echo "🌟 RECOMART V3 ACTIVE (Producer: MANUAL)"
echo "Airflow:   http://3.16.158.217:8080"
echo "MLflow:    http://3.16.158.217:5000"
echo "Dashboard: http://3.16.158.217:8501"
echo ""
echo "👉 TO START LIVE TRAFFIC MANUALLY:"
echo "python3 synthetic_producer.py --speed fast"
echo "------------------------------------------------------"
