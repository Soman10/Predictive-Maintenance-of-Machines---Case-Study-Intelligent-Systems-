import numpy as np
import pandas as pd
import joblib
import json
import tensorflow as tf

class PredictiveMaintenanceInference:
    def __init__(self, artifact_dir="artifacts"):
        self.scaler = joblib.load(f"{artifact_dir}/scaler.joblib") # Update artifact_dir if needed
        self.model = tf.keras.models.load_model(f"{artifact_dir}/failure_ann.keras") # Update artifact_dir if needed

        with open(f"{artifact_dir}/feature_order.json") as f: # Update artifact_dir if needed
            self.feature_order = json.load(f)

        with open(f"{artifact_dir}/type_mapping.json") as f: # Update artifact_dir if needed
            self.type_mapping = json.load(f)

        self.num_features = [
            'Air temperature',
            'Process temperature',
            'Rotational speed',
            'Torque',
            'Tool wear'
        ]

    def preprocess(self, raw_df):
        df = raw_df.copy()

        # Encode Type
        df['Type'] = df['Type'].map(self.type_mapping)
        if df['Type'].isna().any():
            raise ValueError("Invalid machine Type")

        # Scale numeric features
        df[self.num_features] = self.scaler.transform(df[self.num_features])

        return df[self.feature_order].values

    def predict(self, raw_df, threshold=0.5):
        X = self.preprocess(raw_df)
        probs = self.model.predict(X, verbose=0)

        failure_prob = probs[:, 1]
        prediction = (failure_prob >= threshold).astype(int)

        return failure_prob[0], prediction[0]

    def predict_batch(self, raw_df, threshold=0.5):
        X = self.preprocess(raw_df)
        probs = self.model.predict(X, verbose=0)

        return pd.DataFrame({
            "failure_probability": probs[:, 1],
            "failure_prediction": (probs[:, 1] >= threshold).astype(int)
        })

