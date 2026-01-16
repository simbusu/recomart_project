#!/bin/bash

# --- 1. CONFIGURATION ---
export AIRFLOW_HOME=~/recomart_project
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@localhost:5432/airflow
export AIRFLOW__CORE__EXECUTOR=LocalExecutor
export AIRFLOW__CORE__LOAD_EXAMPLES=False
VENV_BIN=$AIRFLOW_HOME/venv/bin

echo "🚀 Starting RecoMart with Docker-backed Model Registry..."

# --- 2. ENVIRONMENT REFRESH ---
cd $AIRFLOW_HOME
source venv/bin/activate

# Ensure we have the MLflow client and Surprise for the DAG
echo -n "📦 Synchronizing Python dependencies... "
pip install "numpy<2.0.0" pandas sqlalchemy nltk mlflow==2.15.1 scikit-surprise -q
echo "✅ Done."

# --- 3. INFRASTRUCTURE (DOCKER) ---
echo -n "🐳 Bringing up Docker Services (Postgres, Kafka, MLflow Registry)... "
# We do NOT use 'down' or '-v' here to protect existing DB data
docker compose up -d > /dev/null 2>&1

# Wait for Postgres to be ready before Airflow tries to migrate
until pg_isready -h localhost -p 5432 -U airflow > /dev/null 2>&1; do 
    printf "."
    sleep 1
done
echo " ✅ Ready."

# --- 4. DATABASE & AIRFLOW PREP ---
echo -n "⚙️ Running Airflow Migrations... "
$VENV_BIN/airflow db migrate > /dev/null 2>&1
$VENV_BIN/python3 init_db.py > /dev/null 2>&1
echo "✅ Done."

# --- 5. START AIRFLOW SERVICES ---
echo "🌪️ Starting Airflow Standalone..."
# We killed the host-side mlflow server here to avoid port 5000 conflict.
# Docker is now handling the MLflow Server.

# Cleanup any zombie Airflow processes first
pkill -f "airflow" > /dev/null 2>&1

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
echo "Airflow UI: http://3.16.158.217:8080"
echo "MLflow Registry: http://3.16.158.217:5000"
echo "------------------------------------------------------"#!/bin/bash

# --- 1. CONFIGURATION ---
export AIRFLOW_HOME=~/recomart_project
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@localhost:5432/airflow
export AIRFLOW__CORE__EXECUTOR=LocalExecutor
export AIRFLOW__CORE__LOAD_EXAMPLES=False
# Fix for Airflow 2.10+ deprecation warning
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@localhost:5432/airflow

# Fix for the [Errno 13] /mlflow Permission Denied error
export MLFLOW_SET_DESTINATION=/home/ubuntu/recomart_project/mlruns
VENV_BIN=$AIRFLOW_HOME/venv/bin

echo "🚀 Starting RecoMart with Docker-backed Model Registry..."

# --- 2. ENVIRONMENT REFRESH ---
cd $AIRFLOW_HOME
source venv/bin/activate

# Ensure we have the MLflow client and Surprise for the DAG
echo -n "📦 Synchronizing Python dependencies... "
pip install "numpy<2.0.0" pandas sqlalchemy nltk mlflow==2.15.1 scikit-surprise -q
echo "✅ Done."

# --- 3. INFRASTRUCTURE (DOCKER) ---
echo -n "🐳 Bringing up Docker Services (Postgres, Kafka, MLflow Registry)... "
# We do NOT use 'down' or '-v' here to protect existing DB data
docker compose up -d > /dev/null 2>&1

# Wait for Postgres to be ready before Airflow tries to migrate
until pg_isready -h localhost -p 5432 -U airflow > /dev/null 2>&1; do 
    printf "."
    sleep 1
done
echo " ✅ Ready."

# --- 4. DATABASE & AIRFLOW PREP ---
echo -n "⚙️ Running Airflow Migrations... "
$VENV_BIN/airflow db migrate > /dev/null 2>&1
$VENV_BIN/python3 init_db.py > /dev/null 2>&1
echo "✅ Done."

# --- 5. START AIRFLOW SERVICES ---
echo "🌪️ Starting Airflow Standalone..."
# We killed the host-side mlflow server here to avoid port 5000 conflict.
# Docker is now handling the MLflow Server.

# Cleanup any zombie Airflow processes first
pkill -f "airflow" > /dev/null 2>&1

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
echo "Airflow UI: http://3.16.158.217:8080"
echo "MLflow Registry: http://3.16.158.217:5000"
echo "------------------------------------------------------"
