import os
import json
import uuid
import requests
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text

# --- LOGGING CONFIGURATION ---
log_file = "ingester.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class RecoMartETL:
    def __init__(self):
        if os.path.exists('/.dockerenv') or os.path.exists('/opt/airflow'):
            self.db_url = "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow"
            self.kafka_host = 'kafka:29092'
            self.base_lake_path = "/opt/airflow/data_lake"
            self.nltk_path = "/opt/airflow/nltk_data"
        else:
            self.db_url = "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"
            self.kafka_host = 'localhost:9092'
            self.base_lake_path = "data_lake"
            self.nltk_path = os.path.expanduser("~/recomart_project/nltk_data")

        self.engine = create_engine(self.db_url)

    # --- NLP & ML METHODS (UNTOUCHED) ---
    def generate_embeddings(self, text_list):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf = TfidfVectorizer(stop_words='english')
            return tfidf.fit_transform(text_list)
        except Exception as e:
            logger.error(f"Embedding Generation Error: {e}")
            return None

    def run_sentiment_analysis(self, text_data):
        try:
            import nltk
            from nltk.sentiment import SentimentIntensityAnalyzer
            if self.nltk_path not in nltk.data.path:
                nltk.data.path.append(self.nltk_path)
            sia = SentimentIntensityAnalyzer()
            return sia.polarity_scores(str(text_data))['compound']
        except Exception as e:
            logger.error(f"Sentiment Analysis Error: {e}")
            return 0.0

    # --- BATCH SYNC METHODS (UPDATED WITH LOGS) ---
    def sync_users_csv(self):
        csv_path = "/opt/airflow/users.csv" if os.path.exists('/.dockerenv') else "users.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # LOGGING THE PULL
            logger.info(f"📊 CSV LOAD: Pulling {len(df)} user records from {csv_path}...")
            
            df.to_sql('users_temp', self.engine, if_exists='replace', index=False)
            with self.engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO users (user_id, name, email, gender, city, signup_date)
                    SELECT user_id, name, email, gender, city, signup_date::DATE
                    FROM users_temp ON CONFLICT (user_id) DO NOTHING
                """))
            logger.info(f"✅ Users synced successfully. Total in batch: {len(df)}")

    def sync_products_api(self, api_url, headers):
        try:
            logger.info(f"🌐 API SYNC: Requesting products from {api_url}...")
            response = requests.get(api_url, headers=headers, timeout=10)
            if response.status_code == 200:
                products = response.json()
                df = pd.DataFrame(products)
                
                # LOGGING THE PULL
                logger.info(f"✅ API LOAD: Successfully pulled {len(df)} products via REST API.")
                
                df.to_sql('products_temp', self.engine, if_exists='replace', index=False)
                with self.engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO products (product_id, product_name, category, price, rating)
                        SELECT product_id, product_name, category, price::REAL, rating::REAL
                        FROM products_temp
                        ON CONFLICT (product_id) DO UPDATE SET
                            product_name = EXCLUDED.product_name, category = EXCLUDED.category,
                            price = EXCLUDED.price, rating = EXCLUDED.rating;
                    """))
                self.update_content_similarity(df)
            else:
                logger.error(f"❌ API Sync Failed: Status Code {response.status_code}")
        except Exception as e:
            logger.error(f"❌ API Sync Error: {e}")

    def update_content_similarity(self, df):
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            if 'product_name' in df.columns:
                df['metadata'] = df['product_name'] + " " + df.get('category', '')
                tfidf_matrix = self.generate_embeddings(df['metadata'])
                if tfidf_matrix is not None:
                    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
                    sim_data = []
                    for idx, row in df.iterrows():
                        similar_indices = sim_matrix[idx].argsort()[-6:-1][::-1]
                        for s_idx in similar_indices:
                            sim_data.append({
                                "product_id": row['product_id'],
                                "similar_product_id": df.iloc[s_idx]['product_id'],
                                "similarity_score": float(sim_matrix[idx][s_idx])
                            })
                    pd.DataFrame(sim_data).to_sql('product_similarity_temp', self.engine, if_exists='replace', index=False)
                    with self.engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO product_similarity_store (product_id, similar_product_id, similarity_score)
                            SELECT product_id, similar_product_id, similarity_score FROM product_similarity_temp
                            ON CONFLICT (product_id, similar_product_id) DO UPDATE SET similarity_score = EXCLUDED.similarity_score
                        """))
                    logger.info(f"✅ Content similarity updated for {len(df)} products.")
        except Exception as e:
            logger.error(f"❌ Similarity Update Error: {e}")

    # --- SURGICAL ADDITION: BEHAVIORAL LINKING ---
    def update_behavioral_similarity(self, user_id, product_id):
        try:
            with self.engine.begin() as conn:
                other_products = conn.execute(text("""
                    SELECT DISTINCT product_id FROM orders
                    WHERE user_id = :user_id AND product_id <> :product_id
                """), {"user_id": user_id, "product_id": product_id}).scalars().all()

                for other_id in other_products:
                    conn.execute(text("""
                        INSERT INTO product_similarity_store (product_id, similar_product_id, similarity_score)
                        VALUES (:p1, :p2, 0.05), (:p2, :p1, 0.05)
                        ON CONFLICT (product_id, similar_product_id)
                        DO UPDATE SET similarity_score = LEAST(product_similarity_store.similarity_score + 0.02, 1.0)
                    """), {"p1": product_id, "p2": other_id})
        except Exception as e:
            logger.debug(f"Behavioral skip: {e}")

    # --- REAL-TIME SPEED LAYER (UPDATED WITH EXPLICIT LOGS) ---
    def land_in_lake(self, data, source_type):
        now = datetime.now()
        path = os.path.join(self.base_lake_path, "kafka", str(source_type),
                            f"year={now.year}", f"month={now.strftime('%m')}", f"day={now.strftime('%d')}")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f"{now.strftime('%H')}_events.jsonl"), "a") as f:
            f.write(json.dumps(data) + "\n")

    def start_speed_layer(self):
        try:
            from kafka import KafkaConsumer
            consumer = KafkaConsumer(
                'updates',
                bootstrap_servers=[self.kafka_host],
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                group_id='recomart-ingester-v7'
            )
            logger.info("🚀 Speed Layer active. Handling Personas & Chaos...")

            for message in consumer:
                data = message.value
                raw_type = data.get('type')
                product_id = data.get('product_id', 'UNKNOWN')
                user_id = data.get('user_id', 'UNKNOWN')

                # --- CHAOS SCRUBBING ---
                try:
                    if raw_type == 'order_transaction' and not isinstance(data.get('total_amount'), (int, float)):
                        data['total_amount'] = float(str(data.get('total_amount')).replace('$','')) 
                    if not raw_type or (not product_id and raw_type != 'click'):
                        raise ValueError("Malformed Kafka Message")
                except Exception as scrub_err:
                    logger.warning(f"☣️ Chaos Scrubbed: {scrub_err}")
                    continue

                # LAND IN LAKE
                self.land_in_lake(data, raw_type)

                # --- EXPLICIT INBOUND LOGS (RESTORED) ---
                if raw_type == 'click':
                    subtype = data.get('event_type', 'view').upper()
                    logger.info(f"📥 [INBOUND] CLICK: {subtype} | User: {user_id} | Prod: {product_id}")
                elif raw_type == 'order_transaction':
                    logger.info(f"💰 [INBOUND] PURCHASE: ${data.get('total_amount')} | User: {user_id} | Prod: {product_id}")
                elif raw_type == 'product_review':
                    logger.info(f"⭐ [INBOUND] REVIEW: {data.get('rating')}/5 | User: {user_id} | Prod: {product_id}")

                # --- DASHBOARD FUNNEL SYNC ---
                try:
                    funnel_type = 'view' if raw_type == 'click' else ('purchase' if raw_type == 'order_transaction' else raw_type)
                    with self.engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO clickstream_logs (event_id, session_id, user_id, product_id, event_type, timestamp)
                            VALUES (:event_id, :session_id, :user_id, :product_id, :event_type, NOW()) ON CONFLICT DO NOTHING
                        """), {
                            "event_id": data.get('event_id') or data.get('order_id') or str(uuid.uuid4()),
                            "session_id": data.get('session_id', 'N/A'),
                            "user_id": user_id, "product_id": product_id, "event_type": funnel_type
                        })
                except Exception: pass

                # --- BUSINESS LOGIC ---
                with self.engine.begin() as conn:
                    if raw_type == 'order_transaction':
                        conn.execute(text("""
                            INSERT INTO orders (order_id, user_id, session_id, order_date, total_amount)
                            VALUES (:order_id, :user_id, :session_id, NOW(), :total_amount) ON CONFLICT DO NOTHING
                        """), data)

                        conn.execute(text("""
                            INSERT INTO product_feature_store (product_id, total_sales, updated_at)
                            VALUES (:product_id, :total_amount, NOW())
                            ON CONFLICT (product_id) DO UPDATE SET
                                total_sales = product_feature_store.total_sales + EXCLUDED.total_sales, updated_at = NOW();
                        """), data)
                        self.update_behavioral_similarity(user_id, product_id)

                    elif raw_type == 'product_review':
                        sentiment = self.run_sentiment_analysis(data.get('review_text', ''))
                        data['sentiment_score'] = sentiment
                        conn.execute(text("""
                            INSERT INTO reviews (review_id, product_id, user_id, rating, review_text, sentiment_score, review_date)
                            VALUES (:review_id, :product_id, :user_id, :rating, :review_text, :sentiment_score, NOW()) ON CONFLICT DO NOTHING
                        """), data)

        except Exception as e:
            logger.error(f"❌ Speed Layer Crash: {e}")

if __name__ == "__main__":
    etl = RecoMartETL()
    etl.sync_users_csv()
    api_url = 'http://localhost:5001/api/products' if not os.path.exists('/.dockerenv') else 'http://host.docker.internal:5001/api/products'
    etl.sync_products_api(api_url, {"X-API-KEY": "recomart_secret_2026"})
    etl.start_speed_layer()
