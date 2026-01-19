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
parser.add_argument('--chaos', type=float, default=0.1, help="Probability of an error event")
args = parser.parse_args()

log_file = "producer.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

TOPIC = 'updates'

if os.path.exists('/.dockerenv'):
    KAFKA_SERVER = 'kafka:29092'
    DB_URL = "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow"
else:
    KAFKA_SERVER = 'localhost:9092'
    DB_URL = "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"

SLEEP_DELAY = 10.0 if args.speed == 'slow' else (0.1, 0.4)

producer = KafkaProducer(
    bootstrap_servers=[KAFKA_SERVER],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# --- SURGICAL ADDITION: CHAOS ENGINE ---
def generate_chaos_event(p_id, u_id):
    chaos_type = random.choice(['missing_fields', 'wrong_types'])
    if chaos_type == 'missing_fields':
        return {"type": "click", "event_id": str(uuid.uuid4()), "note": "corrupt_no_ids"}
    else:
        return {
            "type": "order_transaction",
            "order_id": str(uuid.uuid4()),
            "total_amount": "MALFORMED_PRICE_$$",
            "product_id": p_id,
            "user_id": u_id
        }

# --- 2. SEEDING LOGIC (UNTOUCHED) ---
def seed_reviews(product_ids, user_ids, label="Baseline"):
    logger.info(f"🌟 SEEDING [{label}]: Creating reviews for {len(product_ids)} products...")
    target_count = max(len(product_ids), 10)
    for i in range(target_count):
        p_id = product_ids[i % len(product_ids)]
        u_id = random.choice(user_ids)
        review_event = {
            "type": "product_review",
            "review_id": str(uuid.uuid4()),
            "product_id": p_id,
            "user_id": u_id,
            "rating": float(random.randint(3, 5)),
            "review_text": f"{label} system initialization review.",
            "review_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        producer.send(TOPIC, review_event)
    producer.flush()

# --- 3. DATABASE SYNC & REFRESH (UNTOUCHED) ---
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
                    return list(products), list(users), list(cities), (count == 0)
                time.sleep(10)
        except Exception as e:
            logger.error(f"📡 DB Connection Pending: {e}"); time.sleep(5)

# --- 4. LIVE TRAFFIC (FIXED: INDEPENDENT EVENTS) ---
def run_live_traffic(product_ids, user_ids, cities):
    logger.info(f"🚀 Live traffic active | Speed: {args.speed} | Chaos: {args.chaos*100}%")

    mid_p = len(product_ids) // 2
    cat_alpha = product_ids[:mid_p]
    cat_beta = product_ids[mid_p:]

    pos_txt = ["Great quality", "Excellent!", "Loved it.", "Amazing purchase"]
    neg_txt = ["Bad experience", "Broken", "Waste of money", "Returning this"]

    while True:
        # --- PART A: INDEPENDENT CHAOS ---
        if random.random() < args.chaos:
            bad_event = generate_chaos_event(random.choice(product_ids), random.choice(user_ids))
            producer.send(TOPIC, bad_event)
            logger.warning(f"☣️ Chaos Injected: {bad_event['type']}")

        # --- PART B: STANDARD TRAFFIC (Always runs alongside chaos) ---
        u_id = random.choice(user_ids)
        session_id = str(uuid.uuid4())[:8]
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Persona Logic
        is_alpha_user = int(str(u_id)[-2:]) <= 25
        preferred_cat = cat_alpha if is_alpha_user else cat_beta
        other_cat = cat_beta if is_alpha_user else cat_alpha
        p_id = random.choice(preferred_cat) if random.random() < 0.8 else random.choice(other_cat)
        is_preferred_hit = (p_id in preferred_cat)

        # STEP 1: Clickstream (View)
        producer.send(TOPIC, {
            "type": "click", "event_id": str(uuid.uuid4()), "session_id": session_id,
            "user_id": u_id, "product_id": p_id, "event_type": "view", "timestamp": ts
        })
        logger.info(f"📤 SENT: View | User: {u_id} | Product: {p_id}")

        # STEP 2: Optional Add to Cart
        cart_prob = 0.60 if is_preferred_hit else 0.15
        added_to_cart = False
        if random.random() < cart_prob:
            added_to_cart = True
            producer.send(TOPIC, {
                "type": "click", "event_id": str(uuid.uuid4()), "session_id": session_id,
                "user_id": u_id, "product_id": p_id, "event_type": "add_to_cart", "timestamp": ts
            })

        # STEP 3: Transaction or Review
        event_roll = random.random()
        if (added_to_cart and event_roll < 0.70) or (not added_to_cart and event_roll < 0.05):
            producer.send(TOPIC, {
                "type": "order_transaction", "order_id": str(uuid.uuid4()),
                "session_id": session_id, "product_id": p_id, "user_id": u_id,
                "total_amount": float(round(random.uniform(15.0, 450.0), 2)), "timestamp": ts
            })
            logger.info(f"💰 SENT: Purchase | User: {u_id}")

        elif event_roll > 0.85:
            rating = float(random.randint(4, 5)) if is_preferred_hit else float(random.randint(1, 3))
            txt = random.choice(pos_txt) if rating > 3 else random.choice(neg_txt)
            producer.send(TOPIC, {
                "type": "product_review", "review_id": str(uuid.uuid4()),
                "product_id": p_id, "user_id": u_id, "rating": rating,
                "review_text": txt, "review_date": ts
            })

        time.sleep(random.uniform(*SLEEP_DELAY) if isinstance(SLEEP_DELAY, tuple) else SLEEP_DELAY)

if __name__ == "__main__":
    p_ids, u_ids, cities, is_db_empty = fetch_metadata_and_check_status()
    if is_db_empty: seed_reviews(p_ids, u_ids, label="Baseline")
    run_live_traffic(p_ids, u_ids, cities)
