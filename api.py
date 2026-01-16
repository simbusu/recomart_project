from flask import Flask, request, jsonify
import pandas as pd
import os
import json

app = Flask(__name__)
VALID_KEY = "recomart_secret_2026"

# Ensure path is correct regardless of where script is called
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'products.csv')

@app.route('/api/products', methods=['GET'])
def get_products():
    # 1. Security Check
    if request.headers.get('X-API-KEY') != VALID_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    # 2. Check if file exists
    if not os.path.exists(CSV_PATH):
        return jsonify({"error": "Products source file not found"}), 404

    try:
        # 3. Read and Convert
        df = pd.read_csv(CSV_PATH)
        # We convert to a list of dicts so Flask can return proper JSON
        products = df.to_dict(orient='records')
        return jsonify(products)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Running on 0.0.0.0 allows Docker (host.docker.internal) to see it
    app.run(host='0.0.0.0', port=5001)
