from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

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

        if not req:
            return jsonify({"error": "Request kosong"}), 400

        if isinstance(req, dict):
            req = [req]

        df = pd.DataFrame(req)

        if 'Description' not in df.columns:
            df['Description'] = 'Unknown'

        missing_cols = [c for c in fitur if c not in df.columns]
        if missing_cols:
            return jsonify({
                "error": "Kolom kurang",
                "missing": missing_cols
            }), 400

        for c in fitur:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        if df[fitur].isnull().any().any():
            return jsonify({
                "error": "Ada nilai tidak valid"
            }), 400

        preds = model.predict(df[fitur])

        df_result = pd.DataFrame({
            'Description': df['Description'],
            'prediction': preds
        })

        total_prediction = float(df_result['prediction'].sum())

        top = (
            df_result
            .groupby('Description')['prediction']
            .sum()
            .sort_values(ascending=False)
        )

        top_product = top.index[0] if len(top) > 0 else "Unknown"

        return jsonify({
            "prediction": total_prediction,   # ⭐ INI YANG DIPAKAI LARAVEL
            "top_product": top_product,
            "chart": [
                {"product": k, "qty": float(v)}
                for k, v in top.items()
            ]
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == '__main__':
    app.run(port=5000)