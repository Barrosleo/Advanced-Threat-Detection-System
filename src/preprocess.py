import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    # Perform data cleaning and feature extraction
    data = data.dropna()
    features = data.drop(columns=['label'])
    labels = data['label']
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features, labels

if __name__ == "__main__":
    features, labels = preprocess_data('../data/network_traffic.csv')
    pd.DataFrame(features).to_csv('../data/preprocessed_features.csv', index=False)
    pd.DataFrame(labels).to_csv('../data/preprocessed_labels.csv', index=False)
