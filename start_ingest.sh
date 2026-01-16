#!/bin/bash

# 1. Navigate and Environment
cd ~/recomart_project
source venv/bin/activate

# NEW: Capture speed argument (defaults to fast if not provided)
SPEED_MODE=${1:-fast}

echo "🔄 Initializing Speed Layer (Mode: ${SPEED_MODE})..."

# 2. Structured Partitioning
mkdir -p data_lake/kafka/{user_lifecycle,order_transaction,product_review}

# 3. Cleanup existing processes
echo "🧹 Cleaning up old processes..."
pkill -f "python3 -u ingester.py" || true
pkill -f "python3 synthetic_producer.py" || true

# 4. Start the Ingester (The Consumer / Speed Layer)
echo "📡 Starting Kafka Ingester..."
nohup python3 -u ingester.py > ingester.log 2>&1 &

# 5. Setup Trap
trap "echo -e '\n🛑 Stopping services...'; pkill -f 'python3 -u ingester.py'; pkill -f 'python3 synthetic_producer.py'; exit" INT

echo "✅ Ingester is running (Tail 'ingester.log' for SQL updates)"
echo "⚡ Starting Data Producer simulation..."
echo "------------------------------------------------"
echo "🚀 Traffic Flow: Orders (Speed) | Reviews (Lake) | Profile Updates (Speed)"
echo "📍 Mode: ${SPEED_MODE} | Housekeeping: 48hr Lake Retention Active in DAG"
echo "------------------------------------------------"

# 6. Start the Producer with the dynamic speed flag
# We pass the speed argument directly to the python script
nohup python3 synthetic_producer.py --speed "$SPEED_MODE" > producer_ingest.log 2>&1 &
