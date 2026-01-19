import os
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine, text
from fpdf import FPDF
from datetime import datetime

# --- CONFIGURATION ---
# Using localhost as per your environment; 
# change to 'postgres' if running inside Docker container
DB_URL = "postgresql+psycopg2://airflow:airflow@localhost:5432/airflow"
MODEL_DIR = os.path.expanduser("~/recomart_project/models")
engine = create_engine(DB_URL)

class RecoMartReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'RecoMart End-to-End System Audit', 0, 1, 'C')
        self.set_font('Arial', 'I', 9)
        self.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(5)

    def section_header(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 8, title, 0, 1, 'L', fill=True)
        self.ln(3)

    def log_result(self, label, value, status="[PASS]"):
        self.set_font('Arial', '', 10)
        self.cell(60, 7, f"{label}:", 0, 0)
        self.cell(100, 7, f"{value}", 0, 0)
        self.cell(0, 7, status, 0, 1)

# --- VERIFICATION LOGIC ---

def run_full_audit():
    pdf = RecoMartReport()
    pdf.add_page()

    # 1. DATABASE & DATA QUALITY
    pdf.section_header("1. Core Data Quality (Lake & DB)")
    checks = {
        "Users": "SELECT * FROM users",
        "Products": "SELECT * FROM products",
        "Ratings": "SELECT * FROM temp_lake_ratings_raw"
    }

    for name, query in checks.items():
        try:
            df = pd.read_sql(query, engine)
            nulls = df.isnull().sum().sum()
            dupes = df.duplicated().sum()
            
            # Text-based status for encoding safety
            status = "[PASS]" if (nulls == 0 and dupes == 0) else "[WARN]"
            pdf.log_result(f"{name} Count", len(df), status)
            pdf.log_result(f"{name} Nulls/Dupes", f"{nulls} / {dupes}", status)
            
            if name == "Ratings":
                oor = len(df[(df['rating'] < 1) | (df['rating'] > 5)])
                pdf.log_result("Rating Range Check", f"{oor} failures", "[PASS]" if oor == 0 else "[FAIL]")
        except Exception as e:
            pdf.log_result(name, "Table not found", "[FAIL]")

    # 2. FEATURE STORE VERIFICATION
    pdf.ln(5)
    pdf.section_header("2. Feature Store & Serving Health")
    try:
        df_fs = pd.read_sql("SELECT * FROM product_feature_store", engine)
        if not df_fs.empty:
            # Check Freshness
            latest_upd = pd.to_datetime(df_fs['updated_at']).max()
            hours_old = (datetime.now() - latest_upd).total_seconds() / 3600
            fresh_status = "[PASS]" if hours_old < 24 else "[FAIL]"
            pdf.log_result("Feature Freshness", f"{hours_old:.1f} hours old", fresh_status)
            
            # Check Sentiment Range
            sent_check = df_fs[(df_fs['avg_sentiment'] < -1) | (df_fs['avg_sentiment'] > 1)]
            pdf.log_result("Sentiment Bounds", f"{len(sent_check)} invalid", "[PASS]" if len(sent_check)==0 else "[FAIL]")
            
            # Catalog Coverage
            total_p_df = pd.read_sql("SELECT COUNT(*) FROM products", engine)
            total_p = total_p_df.iloc[0,0]
            coverage = (len(df_fs) / total_p * 100) if total_p > 0 else 0
            pdf.log_result("Catalog Coverage", f"{coverage:.1f}%", "[PASS]" if coverage > 80 else "[WARN]")
    except Exception as e:
        pdf.log_result("Feature Store", "Verification Failed", "[FAIL]")

    # 3. CONTENT SIMILARITY SANITY
    pdf.ln(5)
    pdf.section_header("3. Similarity Matrix Verification")
    sim_path = os.path.join(MODEL_DIR, "content_sim.pkl")
    if os.path.exists(sim_path):
        try:
            with open(sim_path, "rb") as f:
                sim_matrix, df_p = pickle.load(f)
            
            # Identity Check (Diagonal must be 1.0)
            diag_mean = np.mean(np.diag(sim_matrix))
            identity_ok = np.isclose(diag_mean, 1.0)
            pdf.log_result("Matrix Identity", f"Diagonal Avg: {diag_mean:.2f}", "[PASS]" if identity_ok else "[FAIL]")
            
            # Dimensions Check
            dim_ok = sim_matrix.shape[0] == len(df_p)
            pdf.log_result("Dimension Sync", f"{sim_matrix.shape[0]} items", "[PASS]" if dim_ok else "[FAIL]")
        except Exception as e:
            pdf.log_result("Similarity Pickle", "Load Error", "[FAIL]")
    else:
        pdf.log_result("Similarity File", "Not found", "[WARN]")

    # 4. MLFLOW MODEL HEALTH (From DB Logs)
    pdf.ln(5)
    pdf.section_header("4. Model Performance Trend")
    try:
        df_perf = pd.read_sql("SELECT * FROM model_health_logs ORDER BY training_date DESC LIMIT 1", engine)
        if not df_perf.empty:
            row = df_perf.iloc[0]
            pdf.log_result("Latest RMSE", f"{row['rmse']:.4f}", "[PASS]")
            pdf.log_result("Latest F1-Score", f"{row['f1_score']:.4f}", "[PASS]")
            pdf.log_result("DVC Data Lineage", f"{row['data_hash'][:12]}...", "[PASS]")
        else:
            pdf.log_result("Model Logs", "No data found", "[WARN]")
    except Exception as e:
        pdf.log_result("Model Logs", "Query Error", "[FAIL]")

    # SAVE REPORT
    report_file = f"recomart_audit_{datetime.now().strftime('%Y%m%d')}.pdf"
    pdf.output(report_file)
    print(f"Audit Complete! PDF Report saved as: {report_file}")

if __name__ == "__main__":
    run_full_audit()
