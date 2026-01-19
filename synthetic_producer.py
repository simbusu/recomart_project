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
KAFKA_SERVER = 'kafka:29092' if os.path.exists('/.dockerenv') else 'localhost:9092'
DB_URL = "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"

SLEEP_DELAY = 10.0 if args.speed == 'slow' else (0.1, 0.4)

producer = KafkaProducer(
    bootstrap_servers=[KAFKA_SERVER],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# --- 2. SURGICAL FIX: CHAOS ENGINE ---
def generate_chaos_event(p_id, u_id):
    chaos_type = random.choice(['missing_fields', 'wrong_types'])
    if chaos_type == 'missing_fields':
        return {"type": "click", "event_id": str(uuid.uuid4()), "note": "corrupt_no_ids"}
    else:
        return {
            "type": "order_transaction",
            "order_id": str(uuid.uuid4()),
            "session_id": f"CHAOS-{str(uuid.uuid4())[:4]}", # FIX: session_id prevents SQL crash
            "total_amount": "MALFORMED_PRICE_$$",
            "product_id": p_id,
            "user_id": u_id,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

# --- 3. DYNAMIC CSV GROWTH THREAD (Slow Growth) ---
def background_csv_growth(cities):
    """Adds 10 users and 5 products per 24h cycle to the local CSV files."""
    logger.info("🧵 Growth thread active: 24h target cycle.")
    categories = ["Electronics", "Home Office", "Wearables", "Audio", "Gaming"]
    minutes_elapsed = 0
    
    while True:
        time.sleep(60)
        minutes_elapsed += 1

        # ADD USER (Target: ~10 per day)
        if minutes_elapsed % 144 == 0:
            new_uid = f"U{random.randint(100000, 999999)}"
            city = random.choice(cities) if cities else "New York"
            try:
                with open('users.csv', 'a') as f:
                    f.write(f"\n{new_uid},New User,Organic,{city},2026-01-19")
                logger.info(f"👤 [CSV Growth] Added {new_uid}")
            except: pass

        # ADD PRODUCT (Target: ~5 per day)
        if minutes_elapsed % 288 == 0:
            new_pid = f"P{random.randint(100000, 999999)}"
            price = round(random.uniform(29.99, 899.99), 2)
            try:
                with open('products.csv', 'a') as f:
                    f.write(f"\n{new_pid},Growth Item,{random.choice(categories)},RecomArt,{price},4.0")
                logger.info(f"📦 [CSV Growth] Added {new_pid}")
            except: pass

        if minutes_elapsed >= 1440: minutes_elapsed = 0

# --- 4. METADATA REFRESHER (Silent Sync) ---
def run_invisible_refresh(p_ids, u_ids):
    engine = create_engine(DB_URL)
    while True:
        time.sleep(30)
        try:
            with engine.connect() as conn:
                db_p = set(conn.execute(text("SELECT product_id FROM products")).scalars().all())
                db_u = set(conn.execute(text("SELECT user_id FROM users")).scalars().all())
                
                for pid in (db_p - set(p_ids)): p_ids.append(pid)
                for uid in (db_u - set(u_ids)): u_ids.append(uid)
        except: pass

# --- 5. LIVE TRAFFIC LOOP ---
def run_live_traffic(product_ids, user_ids, cities):
    logger.info(f"🚀 Live traffic active | Speed: {args.speed}")
    
    pos_txt = ["Great quality", "Excellent!", "Loved it."]
    neg_txt = ["Bad experience", "Broken", "Waste of money"]

    while True:
        # Chaos Injection
        if random.random() < args.chaos:
            bad_ev = generate_chaos_event(random.choice(product_ids), random.choice(user_ids))
            producer.send(TOPIC, bad_ev)
            logger.warning("☣️ Chaos Injected")

        # Standard Logic
        u_id = random.choice(user_ids)
        p_id = random.choice(product_ids)
        session_id = str(uuid.uuid4())[:8]
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # VIEW
        producer.send(TOPIC, {
            "type": "click", "event_id": str(uuid.uuid4()), "session_id": session_id,
            "user_id": u_id, "product_id": p_id, "event_type": "view", "timestamp": ts
        })

        # PURCHASE (Funnel Logic)
        if random.random() < 0.25:
            producer.send(TOPIC, {
                "type": "order_transaction", "order_id": str(uuid.uuid4()),
                "session_id": session_id, "product_id": p_id, "user_id": u_id,
                "total_amount": float(round(random.uniform(15.0, 450.0), 2)), "timestamp": ts
            })
            logger.info(f"💰 Purchase: {u_id} | Session: {session_id}")

        # REVIEW
        elif random.random() < 0.10:
            rating = float(random.randint(1, 5))
            txt = random.choice(pos_txt) if rating > 3 else random.choice(neg_txt)
            producer.send(TOPIC, {
                "type": "product_review", "review_id": str(uuid.uuid4()),
                "product_id": p_id, "user_id": u_id, "rating": rating,
                "review_text": txt, "review_date": ts
            })

        time.sleep(random.uniform(*SLEEP_DELAY) if isinstance(SLEEP_DELAY, tuple) else SLEEP_DELAY)

if __name__ == "__main__":
    engine = create_engine(DB_URL)
    while True:
        try:
            with engine.connect() as conn:
                p_ids = list(conn.execute(text("SELECT product_id FROM products")).scalars().all())
                u_ids = list(conn.execute(text("SELECT user_id FROM users")).scalars().all())
                cities = list(conn.execute(text("SELECT DISTINCT city FROM users WHERE city IS NOT NULL")).scalars().all())
                if p_ids and u_ids: break
            time.sleep(10)
        except: time.sleep(5)

    threading.Thread(target=run_invisible_refresh, args=(p_ids, u_ids), daemon=True).start()
    threading.Thread(target=background_csv_growth, args=(cities,), daemon=True).start()
    
    run_live_traffic(p_ids, u_ids, cities)
