from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model.pkl')

if not os.path.exists(model_path):
    raise FileNotFoundError("model.pkl tidak ditemukan di project")

with open(model_path, 'rb') as f:
    data = pickle.load(f)

model = data.get('model')
fitur = data.get('fitur')

if model is None or fitur is None:
    raise ValueError("Format model.pkl salah (harus ada 'model' dan 'fitur')")


# =========================
# ROUTE TEST
# =========================
@app.route('/')
def home():
    return "API ML jalan 🚀"


# =========================
# ROUTE PREDICT
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        req = request.get_json()

        # Validasi request kosong
        if not req:
            return jsonify({"error": "Request kosong"}), 400

        # Support single object & list
        if isinstance(req, dict):
            req = [req]

        df = pd.DataFrame(req)

        # Default kolom Description
        if 'Description' not in df.columns:
            df['Description'] = 'Unknown'

        # Cek kolom fitur
        missing_cols = [c for c in fitur if c not in df.columns]
        if missing_cols:
            return jsonify({
                "error": "Kolom kurang",
                "missing": missing_cols
            }), 400

        # Convert ke numerik
        for c in fitur:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # Cek nilai invalid
        if df[fitur].isnull().any().any():
            return jsonify({
                "error": "Ada nilai tidak valid"
            }), 400

        # Prediksi
        preds = model.predict(df[fitur])

        df_result = pd.DataFrame({
            'Description': df['Description'],
            'prediction': preds
        })

        # Total prediksi
        total_prediction = float(df_result['prediction'].sum())

        # Produk teratas
        top = (
            df_result
            .groupby('Description')['prediction']
            .sum()
            .sort_values(ascending=False)
        )

        top_product = top.index[0] if len(top) > 0 else "Unknown"

        # Response
        return jsonify({
            "prediction": total_prediction,
            "top_product": top_product,
            "chart": [
                {"product": str(k), "qty": float(v)}
                for k, v in top.items()
            ]
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# =========================
# RUN SERVER (WAJIB UNTUK RAILWAY)
# =========================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
