import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
import unittest

# Load and preprocess the dataset
def load_and_preprocess_dataset(dataset_path):
    # Read the dataset
    df = pd.read_csv(dataset_path)
    
    # Define features
    numerical_features = ['atemp', 'hum', 'windspeed']
    categorical_features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']

    # Separate the target variable (y) from the features (X)
    X = df[categorical_features + numerical_features]
    y = df['cnt']

    return X, y

# Train and evaluate the model using cross-validation
def train_and_evaluate_model(X, y, n_splits=5):
    # Initialize and train the XGBoost regression model
    model = xgb.XGBRegressor()
    
    # Perform cross-validation
    mads = -cross_val_score(model, X, y, cv=n_splits, scoring='neg_mean_absolute_error')
    
    # Re-train the model on the entire dataset
    model.fit(X, y)
    
    return model, np.mean(mads)

# Predict
def predict(model, X):
    model.predict(X)

# Unit test
class TestPredictionService(unittest.TestCase):
    def test_mad_comparison(self):
        # Load and preprocess the dataset
        dataset_path = 'bike+sharing+dataset/hour.csv'
        X, y = load_and_preprocess_dataset(dataset_path)

        # Train and evaluate the model
        trained_model, mad = train_and_evaluate_model(X, y)

        # Calculate MAD of the data
        mad_data = mean_absolute_error(y, trained_model.predict(X))

        # Calculate the MAD of the prediction and the mean of the data
        mad_mean_data = mean_absolute_error(y, np.mean(y) * np.ones_like(y))

        # Assert that MAD of the prediction and the data is less than MAD of the prediction and the mean of the data
        self.assertLess(mad, mad_mean_data)

def main():
    # Specify the path to the dataset (replace with your dataset)
    dataset_path = 'bike+sharing+dataset/hour.csv'

    # Load and preprocess the dataset
    X, y = load_and_preprocess_dataset(dataset_path)

    # Train and evaluate the model
    trained_model, mad = train_and_evaluate_model(X, y)

    # Print MAD and other model information
    print(f"Mean Absolute Deviation (MAD): {mad}")
    print(f"Trained Model Information:\n{trained_model}")

if __name__ == "__main__":
    main()
    # Run the unit test
    unittest.main()