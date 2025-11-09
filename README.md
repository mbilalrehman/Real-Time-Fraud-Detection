# Real-time Credit Card Fraud Detection (A 2-Part MLOps Project)

This project demonstrates a complete, end-to-end Machine Learning Operations (MLOps) workflow. It is broken into two distinct parts:

1.  **Part 1: The Training Script (`fraud_detection_v2.py`)**
    This script is responsible for loading the raw, imbalanced data, training a powerful AI model, and saving that trained "brain" to disk.
2.  **Part 2: The Prediction API (`app_frontend.py`)**
    This is a separate Flask web application that loads the saved "brain" and serves a real-time API (`/predict`) for a user-facing website.

---

## Project Goal

The goal was to build a robust model to detect fraudulent credit card transactions. The primary challenge was the **extreme class imbalance** (99.8% normal vs. 0.2% fraud), which makes "accuracy" a useless metric.

This project focuses on **F1-Score** and **Recall** for the minority 'Fraud' class.

---

## Part 1: Training the AI (`fraud_detection_v2.py`)

This script performs the classic Data Science workflow.

### Methodology
1.  **Data Ingestion:** Loads the `creditcard.csv` dataset (280,000+ rows) using **Pandas**.
2.  **Preprocessing (Crucial):** Scales the `Time` and `Amount` columns using `StandardScaler`.
3.  **Imbalance Handling (SMOTE):** Applies the **SMOTE (Synthetic Minority Over-sampling Technique)** *only* on the training data. This creates new, synthetic examples of "fraud" to teach the model, balancing the dataset without "leaking" data into the test set.
4.  **Modeling (XGBoost):** Trains a powerful **XGBoost Classifier** on the new, balanced data.
5.  **Evaluation:** Achieves a high F1-Score (approx 0.88+) for the 'Fraud' class on the *original* (imbalanced) test data.

### Output
This script is run *once*. Its only job is to create two "artifact" files:
* `fraud_detection_model.pkl`: The trained XGBoost model (the "brain").
* `scalers.pkl`: The `StandardScaler` objects (the "eyes") needed to process new, unseen data.

---

## Part 2: Serving the AI (`app_frontend.py` & `index.html`)

This is a **Full-Stack Flask Application** that provides a real-time prediction service.

### Features
1.  **MLOps:** On startup, `app_frontend.py` loads the `fraud_detection_model.pkl` and `scalers.pkl` files using `joblib`.
2.  **Frontend:** A simple, "technical demo" frontend (`index.html`) provides a form with 30 input fields (V1-V28, Time, Amount).
3.  **Prediction API:** The form sends data to a `/predict` API endpoint.
4.  **Real-time Preprocessing:** The Flask server takes the raw `Time` and `Amount` from the user, scales them using the loaded `scalers.pkl`, combines them with the V1-V28 features, and sends them to the `model.pkl` for a prediction.
5.  **Result:** The API returns a JSON response ("Normal" or "ALERT: FRAUDULENT") to the user.

---

## How to Run This Project

### Part 1: Train the Model (Run Once)
1.  Create a folder `FraudDetectionProject`.
2.  Place `fraud_detection_v2.py` and `creditcard.csv` inside.
3.  Create a `venv` and `pip install -r requirements.txt` (which includes `xgboost`, `imblearn`, etc.).
4.  Run the script: `python fraud_detection_v2.py`
5.  Wait 1-2 minutes. You will see `model.pkl` and `scalers.pkl` appear in your folder.

### Part 2: Run the Prediction App
1.  Create a new folder `FraudFrontendApp`.
2.  Place `app_frontend.py`, `requirements_frontend.txt`, `model.pkl`, and `scalers.pkl` inside.
3.  Create a `templates` folder and put `index.html` inside it.
4.  Create a new `venv_frontend` and `pip install -r requirements_frontend.txt`.
5.  Run the server: `python app_frontend.py`
6.  Open your browser to `http://127.0.0.1:5000` to use the app.

---

## Key Technologies Used
* **Data Science:** Pandas, Scikit-learn, XGBoost, SMOTE (Imbalanced-learn)
* **Backend:** Python, Flask, Gunicorn
* **MLOps:** Joblib (for model serialization/deserialization)
* **Frontend:** HTML, Tailwind CSS, JavaScript (Fetch API)
