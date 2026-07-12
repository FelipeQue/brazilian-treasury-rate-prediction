import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from typing import Tuple

def split_time_series(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Realiza a divisão entre treino e teste de forma cronológica (80/20)
    utilizando a classe nativa TimeSeriesSplit do scikit-learn.
    Argumentos: X (pd.DataFrame): DataFrame com as variáveis explicativas.
                y (pd.Series): Série com a variável alvo.
    Retorna:    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: DataFrames de treino e teste para X e y.
    """
    time_series_splitter = TimeSeriesSplit(n_splits=4)

    train_indices, test_indices = None, None
    for current_train_index, current_test_index in time_series_splitter.split(X):
        train_indices = current_train_index
        test_indices = current_test_index

    X_train, X_test = X.iloc[train_indices].copy(), X.iloc[test_indices].copy()
    y_train, y_test = y.iloc[train_indices].copy(), y.iloc[test_indices].copy()
    
    return X_train, X_test, y_train, y_test

def calculate_vif(X: pd.DataFrame) -> pd.Series:
    """
    Calcula o VIF de cada coluna de X (DataFrame com as variáveis explicativas de treino).
    Para cada variável, realiza uma regressão linear contra todas as outras,
    captura o R² e aplica a fórmula VIF = 1 / (1 - R²).
    Argumentos: X (pd.DataFrame): DataFrame com as variáveis explicativas.
    Retorna:    pd.Series: Série com os valores de VIF para cada variável, ordenada de forma decrescente.
    """
    vifs = {}
    for column in X.columns:
        y_alvo = X[column]
        X_rest = X.drop(columns=column)
        r2 = LinearRegression().fit(X_rest, y_alvo).score(X_rest, y_alvo)
        if r2 >= 1.0:
            vifs[column] = np.inf
        else:
            vifs[column] = 1 / (1 - r2)      
    return pd.Series(vifs, name="VIF").sort_values(ascending=False)