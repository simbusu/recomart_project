import os
import getpass
import mlflow
import logging

# Configure logging to console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug")

STAGING_DIR = '/home/ubuntu/recomart_project/mlruns'
MLFLOW_URI = "http://172.17.0.1:5000"

print("--- 🔍 MLFLOW ENVIRONMENT DEBUG ---")
try:
    # 1. Check User
    user = getpass.getuser()
    logger.info(f"DEBUG: Running as user: {user}")

    # 2. Set Environment (Just like your DAG does)
    os.makedirs(STAGING_DIR, exist_ok=True)
    os.environ['MLFLOW_SET_DESTINATION'] = STAGING_DIR
    os.environ['MLFLOW_REGISTRY_URI'] = MLFLOW_URI
    
    # 3. Check Write Permissions
    test_file = os.path.join(STAGING_DIR, 'test_write.txt')
    with open(test_file, 'w') as f:
        f.write('test')
    logger.info(f"✅ Debug: Successfully wrote to {STAGING_DIR}")

    # 4. Check MLflow Internal Mapping
    mlflow.set_tracking_uri(MLFLOW_URI)
    logger.info(f"DEBUG: Tracking URI is: {mlflow.get_tracking_uri()}")
    
    # 5. The "Smoking Gun" Check
    # This checks if MLflow is trying to find a configuration file at /mlflow
    logger.info(f"DEBUG: Checking if system root /mlflow exists...")
    if os.path.exists('/mlflow'):
        logger.info("⚠️  The directory /mlflow ALREADY EXISTS on your system root.")
    else:
        logger.info("❌ The directory /mlflow does not exist (this is normal).")

except Exception as e:
    logger.error(f"❌ Debug failed: {e}")
