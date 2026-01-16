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
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RecoMartETL:
    def __init__(self):
        # Determine environment for connection strings
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

    # --- BATCH SYNC METHODS (UNTOUCHED) ---

    def sync_users_csv(self):
        csv_path = "/opt/airflow/users.csv" if os.path.exists('/.dockerenv') else "users.csv"
        if os.path.exists(csv_path):
            logger.info(f"Reading users from {csv_path}...")
            df = pd.read_csv(csv_path)
            df.to_sql('users_temp', self.engine, if_exists='replace', index=False)
            with self.engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO users (user_id, name, email, gender, city, signup_date)
                    SELECT user_id, name, email, gender, city, signup_date::DATE
                    FROM users_temp
                    ON CONFLICT (user_id) DO NOTHING
                """))
            logger.info("✅ Users synced successfully.")

    def sync_products_api(self, api_url, headers):
        try:
            logger.info(f"Calling API: {api_url}")
            response = requests.get(api_url, headers=headers, timeout=10)
            if response.status_code == 200:
                products = response.json()
                df = pd.DataFrame(products)
                df.to_sql('products_temp', self.engine, if_exists='replace', index=False)
                with self.engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO products (product_id, product_name, category, price, rating)
                        SELECT product_id, product_name, category, price::REAL, rating::REAL
                        FROM products_temp
                        ON CONFLICT (product_id) DO UPDATE SET
                            product_name = EXCLUDED.product_name,
                            category = EXCLUDED.category,
                            price = EXCLUDED.price,
                            rating = EXCLUDED.rating;
                    """))
                logger.info(f"✅ Synced {len(products)} products. Updating Similarity Matrix...")
                self.update_content_similarity(df)
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
                    sim_df = pd.DataFrame(sim_data)
                    sim_df.to_sql('product_similarity_temp', self.engine, if_exists='replace', index=False)
                    with self.engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO product_similarity_store (product_id, similar_product_id, similarity_score)
                            SELECT product_id, similar_product_id, similarity_score FROM product_similarity_temp
                            ON CONFLICT (product_id, similar_product_id) DO UPDATE SET
                                similarity_score = EXCLUDED.similarity_score
                        """))
                    logger.info("✅ Similarity scores updated.")
        except Exception as e:
            logger.error(f"❌ Similarity Update Error: {e}")

    # --- REAL-TIME SPEED LAYER (SURGICAL UPDATES ONLY) ---

    def land_in_lake(self, data, source_type):
        now = datetime.now()
        path = os.path.join(self.base_lake_path, "kafka", str(source_type),
                            f"year={now.year}", f"month={now.strftime('%m')}", f"day={now.strftime('%d')}")
        os.makedirs(path, exist_ok=True)
        file_name = f"{now.strftime('%H')}_events.jsonl"
        with open(os.path.join(path, file_name), "a") as f:
            f.write(json.dumps(data) + "\n")

    def start_speed_layer(self):
        try:
            from kafka import KafkaConsumer
            logger.info(f"Connecting to Kafka at {self.kafka_host}...")
            consumer = KafkaConsumer(
                'updates',
                bootstrap_servers=[self.kafka_host],
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                group_id='recomart-ingester-v6'
            )

            logger.info("🚀 Speed Layer active. Monitoring events...")

            for message in consumer:
                data = message.value
                raw_type = data.get('type')
                product_id = data.get('product_id', 'UNKNOWN')

                logger.info(f"📥 INBOUND: {raw_type} | Product: {product_id}")

                # LAND IN LAKE (EXISTING)
                self.land_in_lake(data, raw_type)

                # --- START SURGICAL ADDITION: DASHBOARD FUNNEL SYNC ---
                try:
                    # Map names to match expected Funnel Chart stages
                    funnel_event_type = raw_type
                    if raw_type == 'click': funnel_event_type = 'view'
                    elif raw_type == 'order_transaction': funnel_event_type = 'purchase'

                    with self.engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO clickstream_logs (event_id, session_id, user_id, product_id, event_type, timestamp)
                            VALUES (:event_id, :session_id, :user_id, :product_id, :event_type, NOW())
                            ON CONFLICT (event_id) DO NOTHING
                        """), {
                            "event_id": data.get('event_id') or data.get('order_id') or data.get('review_id') or str(uuid.uuid4()),
                            "session_id": data.get('session_id', 'N/A'),
                            "user_id": data.get('user_id'),
                            "product_id": product_id,
                            "event_type": funnel_event_type
                        })
                except Exception as funnel_err:
                    logger.warning(f"⚠️ Funnel Logging skipped: {funnel_err}")
                # --- END SURGICAL ADDITION ---

                # SPECIFIC BUSINESS LOGIC (EXISTING)
                with self.engine.begin() as conn:
                    if raw_type == 'order_transaction':
                        conn.execute(text("""
                            INSERT INTO orders (order_id, user_id, session_id, order_date, total_amount)
                            VALUES (:order_id, :user_id, :session_id, NOW(), :total_amount)
                            ON CONFLICT (order_id) DO NOTHING
                        """), data)

                        conn.execute(text("""
                            INSERT INTO product_feature_store (product_id, total_sales, updated_at)
                            VALUES (:product_id, :total_amount, NOW())
                            ON CONFLICT (product_id) DO UPDATE SET
                                total_sales = product_feature_store.total_sales + EXCLUDED.total_sales,
                                updated_at = NOW();
                        """), data)

                    elif raw_type == 'product_review':
                        sentiment = self.run_sentiment_analysis(data.get('review_text', ''))
                        data['sentiment_score'] = sentiment

                        conn.execute(text("""
                            INSERT INTO reviews (review_id, product_id, user_id, rating, review_text, sentiment_score, review_date)
                            VALUES (:review_id, :product_id, :user_id, :rating, :review_text, :sentiment_score, NOW())
                            ON CONFLICT (review_id) DO NOTHING
                        """), data)

                        conn.execute(text("""
                            INSERT INTO product_feature_store (product_id, total_sales, review_count, avg_sentiment, updated_at)
                            VALUES (:product_id, 0, 1, :sentiment_score, NOW())
                            ON CONFLICT (product_id) DO UPDATE SET
                                avg_sentiment = (product_feature_store.avg_sentiment * product_feature_store.review_count + EXCLUDED.avg_sentiment) / (product_feature_store.review_count + 1),
                                review_count = product_feature_store.review_count + 1,
                                updated_at = NOW();
                        """), data)

        except Exception as e:
            logger.error(f"❌ Speed Layer Crash: {e}")

if __name__ == "__main__":
    etl = RecoMartETL()
    etl.sync_users_csv()
    api_url = 'http://localhost:5001/api/products' if not os.path.exists('/.dockerenv') else 'http://host.docker.internal:5001/api/products'
    etl.sync_products_api(api_url, {"X-API-KEY": "recomart_secret_2026"})
    etl.start_speed_layer()
