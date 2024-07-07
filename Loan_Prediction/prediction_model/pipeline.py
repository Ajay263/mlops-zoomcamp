# The pipeline can assemble several steps that can be cross-validated together while
# setting different parameters.

# Import Libraries
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# Import other files/modules
from prediction_model.config import config
import prediction_model.processing.preprocessors as pp

# Define the loan processing pipeline
loan_pipe = Pipeline([
    ('Numerical Imputer', pp.NumericalImputer(variables=config.NUMERICAL_FEATURES)),
    ('Categorical Imputer', pp.CategoricalImputer(variables=config.CATEGORICAL_FEATURES)),
    ('Temporal Features', pp.TemporalVariableEstimator(variables=config.TEMPORAL_FEATURES,
                                                       reference_variable=config.TEMPORAL_ADDITION)),
    ('Categorical Encoder', pp.CategoricalEncoder(variables=config.FEATURES_TO_ENCODE)),
    ('Log Transform', pp.LogTransformation(variables=config.LOG_FEATURES)),
    ('Drop Features', pp.DropFeatures(variables_to_drop=config.DROP_FEATURES)),
    ('Scaler Transform', MinMaxScaler()),
    ('Linear Model', LogisticRegression(random_state=1))
])

"""
Pipeline:
    This pipeline processes loan data through several steps:
    1. Numerical Imputer: Fills missing numerical data.
    2. Categorical Imputer: Fills missing categorical data.
    3. Temporal Features: Adds temporal features.
    4. Categorical Encoder: Encodes categorical variables.
    5. Log Transform: Applies log transformation to specified variables.
    6. Drop Features: Drops less significant features.
    7. Scaler Transform: Scales features using MinMaxScaler.
    8. Linear Model: Fits a Logistic Regression model.

Args:
    Numerical Imputer (pp.NumericalImputer): Imputes missing numerical features.
    Categorical Imputer (pp.CategoricalImputer): Imputes missing categorical features.
    Temporal Features (pp.TemporalVariableEstimator): Adds temporal features based on reference variable.
    Categorical Encoder (pp.CategoricalEncoder): Encodes categorical features.
    Log Transform (pp.LogTransformation): Applies log transformation to specified features.
    Drop Features (pp.DropFeatures): Drops specified features.
    Scaler Transform (MinMaxScaler): Scales features to a given range.
    Linear Model (LogisticRegression): Fits a logistic regression model.

Returns:
    Pipeline: A scikit-learn pipeline object.
"""
