#!/bin/bash

# --- 1. CONFIGURATION ---
export AIRFLOW_HOME=~/recomart_project
export VENV_BIN=$AIRFLOW_HOME/venv/bin

# Airflow Connection Strings
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@localhost:5432/airflow
export AIRFLOW__CORE__EXECUTOR=LocalExecutor
export AIRFLOW__CORE__LOAD_EXAMPLES=False

# MLflow Path Fixes (Preventing [Errno 13] /mlflow errors)
export MLFLOW_SET_DESTINATION=/home/ubuntu/recomart_project/mlruns
export MLFLOW_TRACKING_URI=http://172.17.0.1:5000

echo "🚀 Starting RecoMart with Docker-backed Model Registry..."

# --- 2. ENVIRONMENT REFRESH ---
cd $AIRFLOW_HOME
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Please create it first."
    exit 1
fi

echo -n "📦 Synchronizing Python dependencies... "
pip install "numpy<2.0.0" pandas sqlalchemy nltk mlflow==2.15.1 scikit-surprise psycopg2-binary -q
echo "✅ Done."

# --- 3. INFRASTRUCTURE (DOCKER) ---
echo -n "🐳 Bringing up Docker Services (Postgres, Kafka, MLflow)... "
# Restarting containers to ensure fresh volume mounts
docker compose up -d > /dev/null 2>&1

until pg_isready -h localhost -p 5432 -U airflow > /dev/null 2>&1; do
    printf "."
    sleep 1
done
echo " ✅ Ready."

# --- 4. DATABASE & AIRFLOW PREP ---
echo -n "⚙️ Running Airflow Migrations & DB Init... "
$VENV_BIN/airflow db migrate > /dev/null 2>&1
$VENV_BIN/python3 init_db.py > /dev/null 2>&1
echo "✅ Done."

# --- 5. START AIRFLOW SERVICES ---
echo "🌪️ Starting Airflow Standalone..."
# Cleanup zombie processes to prevent port lock
pkill -f "airflow" > /dev/null 2>&1
rm -f $AIRFLOW_HOME/airflow-webserver.pid

nohup $VENV_BIN/airflow standalone > airflow_standalone.log 2>&1 &

echo -n "⏳ Waiting 20s for Web UI... "
sleep 20
echo "✅ Done."

# --- 6. TRIGGER PIPELINE ---
echo -n "🔓 Triggering RecoMart Pipeline... "
$VENV_BIN/airflow dags unpause recomart_full_pipeline > /dev/null 2>&1
$VENV_BIN/airflow dags trigger recomart_full_pipeline > /dev/null 2>&1
echo "✅ Done."

echo "------------------------------------------------------"
echo "🌟 SERVICES ACTIVE"
echo "Airflow UI:      http://3.16.158.217:8080"
echo "MLflow Registry: http://3.16.158.217:5000"
echo "Dashboard:       streamlit run appv1.py"
echo "------------------------------------------------------"
