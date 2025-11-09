# ===== Real-time Fraud Detection (Python Script) =====
# Yeh .ipynb notebook ka .py version hai.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, recall_score
import joblib # Model save karne k liye
import warnings

warnings.filterwarnings('ignore')

def run_fraud_detection():
    
    # --- Section 1: Introduction & Data Loading ---
    print("="*50)
    print("Project: Real-time Fraud Detection (Training Script)")
    print("Goal: Train XGBoost model and save 'model.pkl' AND 'scalers.pkl'")
    print("="*50)

    try:
        df = pd.read_csv('creditcard.csv')
    except FileNotFoundError:
        print("\n!!! ERROR: 'creditcard.csv' file not found!")
        print("Please download and save it in the same folder.")
        return

    print("\n[INFO] Data loaded successfully.")

    # --- Section 2: EDA (Sirf Class distribution) ---
    print("\nClass Distribution (Imbalance Check):")
    print(df['Class'].value_counts())

    # --- Section 3: Data Preprocessing (Scaling & SMOTE) ---
    print("\n--- Section 3: Data Preprocessing ---")
    
    # Scaling: Amount aur Time columns ko scale karein
    # YEH DO (2) SCALERS BOHOT ZAROORI HAIN
    amount_scaler = StandardScaler()
    time_scaler = StandardScaler()
    
    df['scaled_amount'] = amount_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = time_scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    df = df.drop(['Time', 'Amount'], axis=1)
    
    print("[INFO] 'Amount' and 'Time' columns scaled.")

    # X aur y (features aur target) ko define karein
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Train-Test Split (Sab se pehlay)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("[INFO] Data split into Train (80%) and Test (20%).")

    # Handling Imbalance (SMOTE) - SIRF Training Data par
    print("[INFO] Applying SMOTE on Training Data...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    print("New Training Data Class Distribution (After SMOTE):")
    print(y_train_res.value_counts())
    print("[INFO] SMOTE completed.")

    # --- Section 4: Modeling (Sirf XGBoost) ---
    print("\n--- Section 4: Modeling ---")
    
    print("\n[INFO] Training XGBoost Classifier...")
    xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train_res, y_train_res)
    y_pred_xgb = xgb_model.predict(X_test)

    print("\nResults for XGBoost Classifier (on Test Data):")
    print(classification_report(y_test, y_pred_xgb, target_names=['Non-Fraud (0)', 'Fraud (1)']))
    f1_xgb = f1_score(y_test, y_pred_xgb, pos_label=1)
    print(f"Final F1-Score (Fraud Class): {f1_xgb:.4f}")

    # --- Section 5: Save the Model & SCALERS (Sab se Zaroori) ---
    print("\n--- Section 5: Saving the Model & Scalers ---")
    
    # Model (Dimaagh) ko save karein
    model_filename = 'fraud_detection_model.pkl'
    joblib.dump(xgb_model, model_filename)
    print(f"[SUCCESS] Best model (XGBoost) saved as '{model_filename}'")
    
    # Scalers (Aankhein) ko save karein
    scalers = {
        'amount_scaler': amount_scaler,
        'time_scaler': time_scaler
    }
    scalers_filename = 'scalers.pkl'
    joblib.dump(scalers, scalers_filename)
    print(f"[SUCCESS] Scalers saved as '{scalers_filename}'")
    
    print("\nProject (Training) Finished.")

if __name__ == "__main__":
    run_fraud_detection()