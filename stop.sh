#!/bin/bash

# 1. Navigate to project folder
cd ~/recomart_project

echo "🛑 Shutting down RecoMart Lambda System..."

# 2. Stop Background Python Processes
echo "🐍 Killing background Python services..."
# Added -u check and specific script names to be thorough
pkill -f "python3 -u ingester.py" || true
pkill -f "python3 synthetic_producer.py" || true
pkill -f "python3 api.py" || true

# 3. Stop Docker Infrastructure
echo "🐳 Stopping Docker containers (Postgres, Kafka, Airflow)..."
# Using -v is optional, but helps if you want to wipe DB volumes entirely
docker compose down

# 4. Cleanup temporary files and logs
read -p "❓ Do you want to clear ALL logs and the Data Lake? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "🧹 Clearing log files (*.log)..."
    rm -f *.log
    
    if [ -d "data_lake" ]; then
        echo "📂 Wiping and resetting Data Lake partitions..."
        rm -rf data_lake/kafka/*
        # Re-create structure so start.sh/ingester.py don't hit permissions issues
        mkdir -p data_lake/kafka/{user_lifecycle,order_transaction,product_review}
        chmod -R 777 data_lake
        echo "✅ Data Lake structure reset."
    fi
fi

echo "------------------------------------------------"
echo "🏁 System shut down successfully."
echo "💡 To restart fresh, run: ./start.sh"
echo "------------------------------------------------"
