import json
import time
import random
import uuid
import logging
import argparse
import threading
import os
from datetime import datetime
from kafka import KafkaProducer
from sqlalchemy import create_engine, text

# --- 1. CONFIGURATION & ARGS ---
parser = argparse.ArgumentParser()
parser.add_argument('--speed', choices=['slow', 'fast'], default='fast', help="Event frequency")
args = parser.parse_args()

log_file = "producer.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

TOPIC = 'updates'

# Environment-aware connection strings
if os.path.exists('/.dockerenv'):
    KAFKA_SERVER = 'kafka:29092'
    DB_URL = "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow"
else:
    KAFKA_SERVER = 'localhost:9092'
    DB_URL = "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"

# Speed Delay logic
SLEEP_DELAY = 10.0 if args.speed == 'slow' else (0.1, 0.4)

producer = KafkaProducer(
    bootstrap_servers=[KAFKA_SERVER],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# --- 2. SEEDING LOGIC ---
def seed_reviews(product_ids, user_ids, label="Baseline"):
    """Sends seed reviews to Kafka for ML baseline."""
    logger.info(f"🌟 SEEDING [{label}]: Creating reviews for {len(product_ids)} products...")
    target_count = max(len(product_ids), 10) if label == "Baseline" else len(product_ids)

    for i in range(target_count):
        p_id = product_ids[i % len(product_ids)]
        u_id = random.choice(user_ids)
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        review_event = {
            "type": "product_review",
            "review_id": str(uuid.uuid4()),
            "product_id": p_id,
            "user_id": u_id,
            "rating": float(random.randint(3, 5)),
            "review_text": f"{label} system initialization review.",
            "review_date": ts
        }
        producer.send(TOPIC, review_event)
    producer.flush()

# --- 3. DATABASE SYNC & REFRESH ---
def fetch_metadata_and_check_status():
    engine = create_engine(DB_URL)
    while True:
        try:
            with engine.connect() as conn:
                logger.info("🔍 Checking database for products and users...")
                products = conn.execute(text("SELECT product_id FROM products")).scalars().all()
                users = conn.execute(text("SELECT user_id FROM users")).scalars().all()
                cities = conn.execute(text("SELECT DISTINCT city FROM users WHERE city IS NOT NULL")).scalars().all()

                if products and users:
                    count = conn.execute(text("SELECT COUNT(*) FROM product_feature_store")).scalar()
                    is_db_empty = (count == 0)
                    return list(products), list(users), list(cities), is_db_empty

                logger.warning("⏳ DB tables empty. Waiting 10s...")
                time.sleep(10)
        except Exception as e:
            logger.error(f"📡 DB Connection Pending: {e}")
            time.sleep(5)

def run_invisible_refresh(p_ids, u_ids):
    """Background thread to detect new metadata."""
    engine = create_engine(DB_URL)
    while True:
        try:
            time.sleep(30)
            with engine.connect() as conn:
                db_p = set(conn.execute(text("SELECT product_id FROM products")).scalars().all())
                db_u = set(conn.execute(text("SELECT user_id FROM users")).scalars().all())
                new_p = list(db_p - set(p_ids))
                new_u = list(db_u - set(u_ids))

                if new_u:
                    for uid in new_u: u_ids.append(uid)
                    logger.info(f"✨ [Sync] Picked up {len(new_u)} new users.")
                if new_p:
                    seed_reviews(new_p, u_ids, label="New Product")
                    for pid in new_p: p_ids.append(pid)
        except Exception as e:
            logger.error(f"⚠️ Sync Error: {e}")

# --- 4. LIVE TRAFFIC (UPDATED WITH FUNNEL LOGIC) ---
def run_live_traffic(product_ids, user_ids, cities):
    logger.info(f"🚀 Live traffic active | Mode: {args.speed.upper()}")
    pos_txt = ["Great quality", "Excellent!", "Loved it.", "Amazing purchase"]
    neg_txt = ["Bad experience", "Broken", "Waste of money", "Returning this"]

    while True:
        u_id = random.choice(user_ids)
        p_id = random.choice(product_ids)
        session_id = str(uuid.uuid4())[:8]
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # STEP 1: Clickstream (View)
        producer.send(TOPIC, {
            "type": "click",
            "event_id": str(uuid.uuid4()),
            "session_id": session_id,
            "user_id": u_id,
            "product_id": p_id,
            "event_type": "view",
            "timestamp": ts
        })

        # STEP 2: Optional Add to Cart (35% probability)
        # This creates the middle bar in your dashboard funnel
        cart_roll = random.random()
        added_to_cart = False
        if cart_roll < 0.35:
            added_to_cart = True
            producer.send(TOPIC, {
                "type": "click",
                "event_id": str(uuid.uuid4()),
                "session_id": session_id,
                "user_id": u_id,
                "product_id": p_id,
                "event_type": "add_to_cart",
                "timestamp": ts
            })

        # STEP 3: Transaction or Review
        event_roll = random.random()
        
        # Only purchase if they added to cart (realistic) or a small chance of direct buy
        if (added_to_cart and event_roll < 0.60) or (not added_to_cart and event_roll < 0.05):
            producer.send(TOPIC, {
                "type": "order_transaction",
                "order_id": str(uuid.uuid4()),
                "session_id": session_id,
                "product_id": p_id,
                "user_id": u_id,
                "total_amount": float(round(random.uniform(15.0, 450.0), 2)),
                "timestamp": ts
            })
        elif event_roll > 0.85: # Random Review
            rating = float(random.randint(1, 5))
            txt = random.choice(pos_txt) if rating > 3 else random.choice(neg_txt)
            producer.send(TOPIC, {
                "type": "product_review",
                "review_id": str(uuid.uuid4()),
                "product_id": p_id,
                "user_id": u_id,
                "rating": rating,
                "review_text": txt,
                "review_date": ts
            })

        if isinstance(SLEEP_DELAY, tuple):
            time.sleep(random.uniform(*SLEEP_DELAY))
        else:
            time.sleep(SLEEP_DELAY)

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        p_ids, u_ids, cities, is_db_empty = fetch_metadata_and_check_status()
        if is_db_empty:
            seed_reviews(p_ids, u_ids, label="Baseline")

        threading.Thread(target=run_invisible_refresh, args=(p_ids, u_ids), daemon=True).start()
        run_live_traffic(p_ids, u_ids, cities)

    except KeyboardInterrupt:
        logger.info("🛑 Producer stopped.")
