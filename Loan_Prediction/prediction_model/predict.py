# This module loads the saved ML model (.pkl) and makes predictions on the
# new data.

# Import Libraries
import pandas as pd
import numpy as np
import joblib

# Import other files/modules
from prediction_model.config import config
from prediction_model.processing.data_management import load_pipeline

# Load the trained pipeline
pipeline_file_name = 'classification_v1.pkl'
loan_pipe = load_pipeline(pipeline_file_name)

def make_prediction(input_data):
    """Predicts the output using the trained pipeline.

    Args:
        input_data (dict or DataFrame): The input data for making predictions.

    Returns:
        dict: A dictionary containing the predictions.
    """
    
    # Read Data
    data = pd.DataFrame(input_data)
    
    # Make prediction
    prediction = loan_pipe.predict(data[config.FEATURES])
    output = np.where(prediction == 1, 'Y', 'N').tolist()
    results = {'prediction': output}
    
    return results
