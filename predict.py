"""
Reusable Prediction Engine for Airline Passenger Satisfaction
Loads the trained Scikit-Learn pipeline and serves single & batch inference.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_model_pipeline.joblib')
METADATA_PATH = os.path.join(os.path.dirname(__file__), 'models', 'feature_metadata.json')

# Cache for pipeline and metadata in-memory
_cached_pipeline = None
_cached_metadata = None


def get_metadata():
    """Loads and caches feature metadata."""
    global _cached_metadata
    if _cached_metadata is None:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                _cached_metadata = json.load(f)
        else:
            _cached_metadata = {
                'all_features': [
                    'Age', 'Flight Distance', 'Inflight wifi service',
                    'Departure/Arrival time convenient', 'Ease of Online booking',
                    'Gate location', 'Food and drink', 'Online boarding',
                    'Seat comfort', 'Inflight entertainment', 'On-board service',
                    'Leg room service', 'Baggage handling', 'Checkin service',
                    'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes',
                    'Gender', 'Customer Type', 'Type of Travel', 'Class'
                ],
                'categorical_features': ['Gender', 'Customer Type', 'Type of Travel', 'Class'],
                'target_classes': ['Dissatisfied', 'Satisfied']
            }
    return _cached_metadata


def get_pipeline(model_path=MODEL_PATH):
    """Loads and caches the trained scikit-learn pipeline."""
    global _cached_pipeline
    if _cached_pipeline is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model pipeline not found at '{model_path}'. Please run 'python train.py' first."
            )
        _cached_pipeline = joblib.load(model_path)
    return _cached_pipeline


def validate_features(df, metadata=None):
    """
    Validates that the input DataFrame contains all required feature columns.
    Returns (is_valid, missing_columns_list).
    """
    if metadata is None:
        metadata = get_metadata()
    required = set(metadata['all_features'])
    present = set(df.columns)
    missing = list(required - present)
    return len(missing) == 0, missing


def predict_single(passenger_data: dict, pipeline=None):
    """
    Predict satisfaction for a single passenger.
    
    Args:
        passenger_data (dict): Dictionary with feature names and values.
        pipeline: Optional pre-loaded pipeline instance.
        
    Returns:
        dict: Detailed prediction results with probabilities and confidence.
    """
    if pipeline is None:
        pipeline = get_pipeline()
    metadata = get_metadata()

    # Convert to single-row DataFrame
    df = pd.DataFrame([passenger_data])

    # Validate or fill missing feature columns with sensible defaults
    is_valid, missing = validate_features(df, metadata)
    if not is_valid:
        raise ValueError(f"Input passenger data is missing required features: {missing}")

    # Ensure correct feature ordering
    df = df[metadata['all_features']]

    # Model inference
    class_pred = pipeline.predict(df)[0]
    probs = pipeline.predict_proba(df)[0]

    prob_dissatisfied = float(probs[0])
    prob_satisfied = float(probs[1])

    label = 'Satisfied' if class_pred == 1 else 'Dissatisfied'
    confidence = prob_satisfied if class_pred == 1 else prob_dissatisfied

    return {
        'prediction_class': int(class_pred),
        'prediction_label': label,
        'probability_satisfied': round(prob_satisfied, 4),
        'probability_dissatisfied': round(prob_dissatisfied, 4),
        'confidence_percentage': round(confidence * 100, 2)
    }


def predict_batch(data, pipeline=None):
    """
    Performs batch inference on a DataFrame or CSV filepath.
    
    Args:
        data (pd.DataFrame or str): Input DataFrame or path to CSV file.
        pipeline: Optional pre-loaded pipeline.
        
    Returns:
        pd.DataFrame: Original records augmented with predictions and probabilities.
    """
    if pipeline is None:
        pipeline = get_pipeline()
    metadata = get_metadata()

    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("Input data must be a pandas DataFrame or file path string.")

    is_valid, missing = validate_features(df, metadata)
    if not is_valid:
        raise ValueError(
            f"Input dataset is missing the following {len(missing)} required columns: {missing}"
        )

    # Feature matrix in exact training order
    X = df[metadata['all_features']]

    # Predict
    class_preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)

    output_df = df.copy()
    output_df['Predicted_Satisfaction'] = np.where(class_preds == 1, 'Satisfied', 'Dissatisfied')
    output_df['Probability_Satisfied'] = np.round(probs[:, 1], 4)
    output_df['Probability_Dissatisfied'] = np.round(probs[:, 0], 4)
    output_df['Confidence'] = np.round(np.max(probs, axis=1) * 100, 2)

    return output_df


if __name__ == '__main__':
    print("Testing Airline Passenger Satisfaction Prediction Engine...")

    # Sample Profile 1: Frequent Business Traveler (High satisfaction expected)
    satisfied_sample = {
        'Gender': 'Female',
        'Customer Type': 'Loyal Customer',
        'Age': 42,
        'Type of Travel': 'Business travel',
        'Class': 'Business',
        'Flight Distance': 1850,
        'Inflight wifi service': 5,
        'Departure/Arrival time convenient': 4,
        'Ease of Online booking': 4,
        'Gate location': 4,
        'Food and drink': 4,
        'Online boarding': 5,
        'Seat comfort': 5,
        'Inflight entertainment': 5,
        'On-board service': 5,
        'Leg room service': 5,
        'Baggage handling': 5,
        'Checkin service': 5,
        'Cleanliness': 5,
        'Departure Delay in Minutes': 0,
        'Arrival Delay in Minutes': 0.0
    }

    # Sample Profile 2: Personal Economy Traveler with delays (Dissatisfaction expected)
    dissatisfied_sample = {
        'Gender': 'Male',
        'Customer Type': 'disloyal Customer',
        'Age': 23,
        'Type of Travel': 'Personal Travel',
        'Class': 'Eco',
        'Flight Distance': 450,
        'Inflight wifi service': 1,
        'Departure/Arrival time convenient': 2,
        'Ease of Online booking': 1,
        'Gate location': 2,
        'Food and drink': 1,
        'Online boarding': 1,
        'Seat comfort': 2,
        'Inflight entertainment': 1,
        'On-board service': 2,
        'Leg room service': 1,
        'Baggage handling': 2,
        'Checkin service': 2,
        'Cleanliness': 1,
        'Departure Delay in Minutes': 65,
        'Arrival Delay in Minutes': 75.0
    }

    res1 = predict_single(satisfied_sample)
    print("\n[Sample 1 - Business Traveler]")
    print(f"Result: {res1['prediction_label']} (Confidence: {res1['confidence_percentage']}%, P(Satisfied)={res1['probability_satisfied']})")

    res2 = predict_single(dissatisfied_sample)
    print("\n[Sample 2 - Economy Traveler with Delays]")
    print(f"Result: {res2['prediction_label']} (Confidence: {res2['confidence_percentage']}%, P(Satisfied)={res2['probability_satisfied']})")

    # Test Batch Inference with top 10 rows of test.csv
    print("\nTesting Batch Inference on test.csv subset...")
    test_head = pd.read_csv('test.csv', nrows=10)
    batch_res = predict_batch(test_head)
    print(batch_res[['id', 'Class', 'Type of Travel', 'Predicted_Satisfaction', 'Probability_Satisfied', 'Confidence']])
    print("\nPrediction engine test completed successfully!")
