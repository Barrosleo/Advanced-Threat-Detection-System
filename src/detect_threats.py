import joblib
import pandas as pd
from preprocess import preprocess_data

def detect_threats(log_file, model_path):
    model = joblib.load(model_path)
    features, _ = preprocess_data(log_file)
    predictions = model.predict(features)
    threats = features[predictions == 1]
    return threats

if __name__ == "__main__":
    threats = detect_threats('../logs/realtime_log.csv', '../models/threat_detection_model.pkl')
    if not threats.empty:
        print("Threats detected:")
        print(threats)
    else:
        print("No threats detected.")
