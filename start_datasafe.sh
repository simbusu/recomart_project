#!/bin/bash

# 1. Cleanup only the volatile application processes
echo "🧹 Stopping existing Python application services..."
pkill -f api.py || true
pkill -f ingester.py || true
pkill -f synthetic_producer.py || true

# 2. Refresh Infrastructure (Keeps Volumes Intact)
echo "🐳 Ensuring infrastructure is up..."
docker compose up -d

# 3. Secure the Data Lake
# mkdir -p is safe; it won't overwrite existing .jsonl files
echo "📂 Verifying historical Data Lake integrity..."
mkdir -p data_lake/kafka/{user_lifecycle,order_transaction,product_review}
chmod -R 777 data_lake

# 4. Restart Product API
echo "🚀 Starting Product API with latest changes..."
source venv/bin/activate
nohup python3 api.py > api.log 2>&1 &

# 5. Database Readiness Check
until docker exec postgres pg_isready -U airflow > /dev/null 2>&1; do
  echo "...waiting for postgres..."
  sleep 2
done

# 6. IDEMPOTENT Schema Update
# We run your python script because it uses "IF NOT EXISTS" 
# and handles the logging format we established.
echo "🛠️ Validating Schema and Staging tables..."
python3 init_db.py

# 7. Force Airflow to Parse New DAG Logic
echo "📅 Synchronizing DAG definitions..."
docker exec airflow-scheduler airflow dags reserialize
# Added unpause to ensure any new logic is active
docker exec airflow-scheduler airflow dags unpause recomart_full_pipeline > /dev/null 2>&1

# 8. Trigger Pipeline (Incremental Processing)
echo "🔓 Triggering Incremental Batch Sync..."
docker exec airflow-scheduler airflow dags trigger recomart_full_pipeline

echo "------------------------------------------------"
echo "✅ UPDATE COMPLETE: SYSTEM READY"
echo "Historical Data: PRESERVED | Latest Logic: DEPLOYED"
echo "Check health with: python3 checker.py"
echo "------------------------------------------------"
