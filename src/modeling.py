import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import RegressorMixin

# TODO Importar constantes de configuração

"""
Treino, avaliação e persistência do modelo de regressão linear (Fases 5 e 6).
"""

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
    """Treina um modelo de regressão nos dados de treino."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: RegressorMixin, X: pd.DataFrame, y_true: pd.Series) -> dict:
    """Calcula MAE, MSE, RMSE e R2 das previsões do modelo."""
    y_pred = model.predict(X)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }

