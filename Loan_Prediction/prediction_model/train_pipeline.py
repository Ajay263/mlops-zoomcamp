# This module loads the training data and passes it to the pipeline, then saves the
# pickle file of the model to the local directory
#Import Libraries
import pandas as pd
import numpy as np

#Import other files/modules
from config import config
from prediction_model.processing.data_management import load_dataset, save_pipeline
import prediction_model.processing.preprocessors as pp
import prediction_model.pipeline as pl
from predict import make_prediction

def run_training():
    """Train the model"""

    #Read Data
    train = load_dataset(config.TRAIN_FILE)

    #separating Loan_status in y
    y = train[config.TARGET].map({'N':0 , 'Y':1})
    pl.loan_pipe.fit(train[config.FEATURES],y)
    save_pipeline(pipeline_to_save=pl.loan_pipe)

if __name__=='__main__':
    run_training()