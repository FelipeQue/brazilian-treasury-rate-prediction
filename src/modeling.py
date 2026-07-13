import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# TODO Importar constantes de configuração

"""
Treino, avaliação e persistência do modelo de regressão linear (Fases 5 e 6).
"""

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Treina uma Regressão Linear simples nos dados de treino."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

