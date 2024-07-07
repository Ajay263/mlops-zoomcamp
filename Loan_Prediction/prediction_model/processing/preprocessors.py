# module for the fit and transform functions required by the sklearn pipeline

# Import Libraries
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

# Import other files/modules
from prediction_model.config import config

# Numerical Imputer
class NumericalImputer(BaseEstimator, TransformerMixin):
    """Numerical Data Missing Value Imputer"""
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        """
        Fit the imputer on the dataset.

        Args:
            X (pd.DataFrame): The input dataframe.
            y (None): Not used, present for API consistency by convention.

        Returns:
            self: Returns self.
        """
        self.imputer_dict_ = {}
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mean()
        return self

    def transform(self, X):
        """
        Transform the dataset by filling missing values with the mean.

        Args:
            X (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The transformed dataframe with missing values imputed.
        """
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X

# Categorical Imputer
class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Categorical Data Missing Value Imputer"""
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        """
        Fit the imputer on the dataset.

        Args:
            X (pd.DataFrame): The input dataframe.
            y (None): Not used, present for API consistency by convention.

        Returns:
            self: Returns self.
        """
        self.imputer_dict_ = {}
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mode()[0]
        return self

    def transform(self, X):
        """
        Transform the dataset by filling missing values with the mode.

        Args:
            X (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The transformed dataframe with missing values imputed.
        """
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X

# Categorical Encoder
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Categorical Data Encoder"""
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y):
        """
        Fit the encoder on the dataset.

        Args:
            X (pd.DataFrame): The input dataframe.
            y (pd.Series): The target variable (not used).

        Returns:
            self: Returns self.
        """
        self.encoder_dict_ = {}
        for var in self.variables:
            t = X[var].value_counts().sort_values(ascending=True).index
            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}
        return self

    def transform(self, X):
        """
        Transform the dataset by encoding categorical variables.

        Args:
            X (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The transformed dataframe with encoded categorical variables.
        """
        X = X.copy()
        # This part assumes that the encoder does not introduce NANs
        # In that case, a check needs to be done and the code should break
        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])
        return X

# Temporal Variables
class TemporalVariableEstimator(BaseEstimator, TransformerMixin):
    """Feature Engineering"""
    def __init__(self, variables=None, reference_variable=None):
        self.variables = variables
        self.reference_variable = reference_variable

    def fit(self, X, y=None):
        """
        Fit the estimator on the dataset.

        Args:
            X (pd.DataFrame): The input dataframe.
            y (None): Not used, present for API consistency by convention.

        Returns:
            self: Returns self.
        """
        # No need to put anything, needed for Sklearn Pipeline
        return self

    def transform(self, X):
        """
        Transform the dataset by adding temporal features.

        Args:
            X (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The transformed dataframe with added temporal features.
        """
        X = X.copy()
        for var in self.variables:
            X[var] = X[var] + X[self.reference_variable]
        return X

# Log Transformations
class LogTransformation(BaseEstimator, TransformerMixin):
    """Transforming variables using Log Transformations"""
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y):
        """
        Fit the transformer on the dataset.

        Args:
            X (pd.DataFrame): The input dataframe.
            y (pd.Series): The target variable (not used).

        Returns:
            self: Returns self.
        """
        return self

    def transform(self, X):
        """
        Transform the dataset by applying log transformations.

        Args:
            X (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The transformed dataframe with log-transformed variables.
        """
        # Need to check in advance if the features are <= 0
        # If yes, needs to be transformed properly (E.g., np.log1p(X[var]))
        X = X.copy()
        for var in self.variables:
            X[var] = np.log(X[var])
        return X

# Drop Features
class DropFeatures(BaseEstimator, TransformerMixin):
    """Dropping Features Which Are Less Significant"""
    def __init__(self, variables_to_drop=None):
        self.variables_to_drop = variables_to_drop

    def fit(self, X, y=None):
        """
        Fit the transformer on the dataset.

        Args:
            X (pd.DataFrame): The input dataframe.
            y (None): Not used, present for API consistency by convention.

        Returns:
            self: Returns self.
        """
        return self

    def transform(self, X):
        """
        Transform the dataset by dropping specified features.

        Args:
            X (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The transformed dataframe with specified features dropped.
        """
        X = X.copy()
        X = X.drop(self.variables_to_drop, axis=1)
        return X
