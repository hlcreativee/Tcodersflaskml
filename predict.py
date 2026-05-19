from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model.pkl')

with open(model_path, 'rb') as f:
    data = pickle.load(f)

model = data['model']
fitur = data['fitur']


@app.route('/predict', methods=['POST'])
def predict():
    try:
        req = request.get_json()

        # Validasi input
        if not req:
            return jsonify({"error": "Request kosong"}), 400

        if isinstance(req, dict):
            req = [req]

        df = pd.DataFrame(req)

        # Pastikan ada kolom Description
        if 'Description' not in df.columns:
            df['Description'] = 'Unknown'

        # Cek fitur
        missing_cols = [col for col in fitur if col not in df.columns]
        if missing_cols:
            return jsonify({
                "error": "Kolom kurang",
                "missing": missing_cols
            }), 400

        # Convert numeric
        for col in fitur:
            df[col] = pd.to_numeric(df[col], errors='coerce')

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

        # Ambil TOP PRODUK
        hasil = (
            df_result
            .groupby('Description')['prediction']
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )

        return jsonify({
            "top_product": hasil.index[0],
            "chart": [
                {"product": k, "qty": float(v)}
                for k, v in hasil.items()
            ]
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == '__main__':
    app.run(port=5000)