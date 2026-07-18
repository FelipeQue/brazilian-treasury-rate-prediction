import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.config import MODEL_DIR

"""
Treino, avaliação e persistência do modelo de regressão linear, ridge e lasso (Fases 5 e 6).
"""

_MODEL_MAPPING = {
    "linear": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "xgboost": XGBRegressor
}

def train_model(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str = "linear",
        **kwargs
        ) -> RegressorMixin:
    """Treina um modelo de regressão nos dados de treino."""
    model_type_lower = model_type.lower()
    if model_type_lower not in _MODEL_MAPPING:
        raise ValueError(f"Tipo de modelo inválido: {model_type}. Escolha entre 'linear', 'ridge' ou 'lasso'.")
    model_class = _MODEL_MAPPING[model_type_lower]
    model = model_class(**kwargs)
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

def print_metrics(metrics: dict, title: str) -> None:
    """Imprime no notebook as métricas de avaliação do modelo."""
    print(title)
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:,.4f}")

def save_model(model: RegressorMixin | dict, version_folder: str, filename: str) -> None:
    """Salva o modelo treinado em um arquivo joblib dentro da pasta MODEL_DIR."""
    (MODEL_DIR / version_folder).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / version_folder / filename)
    print(f"Modelo salvo em: {MODEL_DIR / version_folder / filename}")

def save_metrics(metrics: dict, version_folder: str, filename: str) -> None:
    """Salva o dicionário de metadados/métricas dentro da pasta MODEL_DIR."""
    (MODEL_DIR / version_folder).mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / version_folder / filename, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Métricas salvas em: {MODEL_DIR / version_folder / filename}")