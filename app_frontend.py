# ===== Fraud Prediction API (Flask) =====
# Yeh model ko load kar k user ko prediction dega

import numpy as np
from flask import Flask, request, render_template, jsonify
import joblib

app = Flask(__name__, template_folder='templates')

# --- Model aur Scalers ko Load Karna ---
try:
    # Model (Dimaagh) ko load karein
    model = joblib.load('fraud_detection_model.pkl')
    # Scalers (Aankhein) ko load karein
    scalers = joblib.load('scalers.pkl')
    amount_scaler = scalers['amount_scaler']
    time_scaler = scalers['time_scaler']
    print("[INFO] Model and scalers loaded successfully!")
except FileNotFoundError:
    print("[ERROR] 'fraud_detection_model.pkl' ya 'scalers.pkl' nahi mila!")
    print("[ERROR] Please run 'fraud_detection_v2.py' first!")
    model = None
    amount_scaler = None
    time_scaler = None

# --- Routes (Web Pages) ---

@app.route('/')
def home():
    """Main page (index.html) ko render karna."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Form se data le kar prediction karna."""
    if not model:
        return jsonify({'error': 'Model is not loaded. Please check server logs.'}), 500
        
    try:
        # Form se saara data strings k taur par lein
        # 30 features (V1-V28, Time, Amount)
        v_features = [float(request.form[f'v{i}']) for i in range(1, 29)]
        time_val = float(request.form['time'])
        amount_val = float(request.form['amount'])
        
        # --- Preprocessing (Wahi scaling jo training mein ki thi) ---
        # 1. Scalers ko 2D array chahiye
        scaled_time = time_scaler.transform(np.array([[time_val]]))
        scaled_amount = amount_scaler.transform(np.array([[amount_val]]))
        
        # 2. Saaray features ko ek single list mein combine karein
        # (V1...V28, scaled_amount, scaled_time) - Yahi order training mein tha
        feature_list = v_features + [scaled_amount[0][0], scaled_time[0][0]]
        
        # 3. Model ko prediction k liye 2D array dein
        final_features = np.array(feature_list).reshape(1, -1)
        
        # --- Prediction Karna ---
        prediction = model.predict(final_features)
        probability = model.predict_proba(final_features)[0][1] # Fraud ki probability
        
        # Result ko frontend ko bhejein
        if prediction[0] == 1:
            result_text = "ALERT: FRAUDULENT Transaction"
            probability_text = f"Probability of Fraud: {probability*100:.2f}%"
        else:
            result_text = "Normal Transaction"
            probability_text = f"Probability of Fraud: {probability*100:.2f}%"
            
        return jsonify({
            'result': result_text,
            'probability': probability_text
        })
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {str(e)}")
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)