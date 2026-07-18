import pandas as pd
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from typing import Tuple

"""
Limpeza, engenharia de atributos e preparação para modelagem (Fases 2, 3 e 4 do notebook).
"""

def calculate_iqr_bounds(series, factor=1.5):
    """
    Função auxiliar para calcular os limites inferior e superior usando o método IQR.
    Argumentos: series (pd.Series): Série numérica.
                factor (float): Fator multiplicador do IQR para definir os limites (padrão: 1.5).
    Retorna:    tuple: Limite inferior e limite superior.
    """
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (factor * iqr)
    upper_bound = q3 + (factor * iqr)
    return lower_bound, upper_bound

def treat_outliers_iqr(df, columns, factor=1.5, method="remove"):
    """
    Trata outliers numéricos usando o método IQR, com opção de remoção ou limitação dos valores. O parâmetro `logarithmic` permite aplicar uma transformação logarítmica aos dados antes de calcular os limites do IQR, o que pode ser útil para colunas com distribuição altamente assimétrica.
    Argumentos: df (pd.DataFrame): DataFrame a ser tratado.
                columns (list): Lista de colunas numéricas a serem tratadas.
                factor (float): Fator multiplicador do IQR para definir os limites (padrão: 1.5).
                method (str): Método de tratamento dos outliers: 'remove' para remover registros ou 'limit' para limitar os valores aos limites do IQR.
    Retorna:    pd.DataFrame: DataFrame com os outliers tratados.
    """
    df = df.copy()
    bounds = {}

    for column in columns:
        lower_bound, upper_bound = calculate_iqr_bounds(df[column], factor)            
        bounds[column] = (lower_bound, upper_bound)
        is_outlier = (df[column] < lower_bound) | (df[column] > upper_bound)
        
        print(
            f"Coluna '{column}': Encontrados {is_outlier.sum()} outliers "
            f"(limite_inferior={lower_bound:.2f}, limite_superior={upper_bound:.2f})"
        )

    for column in columns:
        lower_bound, upper_bound = bounds[column]
        if method == "remove":
            is_valid = (df[column] >= lower_bound) & (df[column] <= upper_bound)
            df = df[is_valid]
        elif method == "limit":
            df.loc[df[column] < lower_bound, column] = lower_bound
            df.loc[df[column] > upper_bound, column] = upper_bound
            
    return df

def add_duration_column(df: pd.DataFrame, date: str = 'DATA', maturity_date: str = 'VENCIMENTO') -> pd.DataFrame:
    """
    Adiciona uma nova coluna 'DURACAO' que representa a duração em dias entre a data do leilão e a data de vencimento do título.
    Argumentos: df (pd.DataFrame): DataFrame contendo as colunas de datas.
                date (str): Nome da coluna com a data do leilão.
                maturity_date (str): Nome da coluna com a data de vencimento.
    Retorna:    pd.DataFrame: DataFrame com a nova coluna 'DURACAO'.
    """
    df = df.copy()
    df['DURACAO'] = (df[maturity_date] - df[date]).dt.days
    return df

def add_market_rejection_column(df: pd.DataFrame, accepted_column: str = 'ACEITO/OFERTADO', threshold: float = 0.15) -> pd.DataFrame:
    """
    Adiciona uma nova coluna booleana (mas com tipo int, para ser usada em modelos de machine learning) 'REJEICAO_MERCADO' que indica se o leilão sofreu forte rejeição/frustração por parte dos dealers do mercado.
    Argumentos: df (pd.DataFrame): DataFrame contendo a coluna de proporção aceito/ofertado.
                accepted_column (str): Nome da coluna com a razão aceito/ofertado.
                threshold (float): Limiar abaixo do qual o leilão é considerado rejeitado (padrão: 15%).
    Retorna:    pd.DataFrame: DataFrame modificado com a nova coluna 'REJEICAO_MERCADO'.
    """
    df = df.copy()
    df['REJEICAO_MERCADO'] = (df[accepted_column] < threshold).astype(int)
    return df

def add_lag_column(df: pd.DataFrame, date: str = 'DATA', maturity: str = 'VENCIMENTO', target_column: str = 'TAXA') -> pd.DataFrame:
    """
    Ordena o dataset cronologicamente por título e adiciona a feature de lag 
    (taxa do leilão anterior daquele vencimento específico).
    Remove as linhas iniciais de cada título que conterão valores nulos (NaN).
    Argumentos: df (pd.DataFrame): DataFrame original.
                date (str): Nome da coluna de data do leilão.
                target_column (str): Nome da variável alvo (target) para calcular o lag.
    Retorna:    pd.DataFrame: DataFrame ordenado, com a nova coluna e sem valores nulos no lag.
    """
    df = df.copy()
    df[maturity] = pd.to_datetime(df[maturity])
    df[date] = pd.to_datetime(df[date])
    df = df.sort_values(by=[maturity, date]).reset_index(drop=True)
    df['TAXA_ULTIMO_LEILAO'] = df.groupby(maturity)[target_column].shift(1)
    df = df.dropna(subset=['TAXA_ULTIMO_LEILAO']).reset_index(drop=True)
    df = df.sort_values(by=date).reset_index(drop=True)
    return df

def select_final_columns(df: pd.DataFrame, feature_columns: list[str], target_column: str) -> pd.DataFrame:
    """
    Seleciona apenas as variáveis explicativas e o alvo: o recorte de colunas que efetivamente entra na modelagem.
    Argumentos: df (pd.DataFrame): DataFrame processado.
                feature_columns (list): Lista de nomes das colunas de variáveis explicativas.
                target_column (str): Nome da coluna da variável alvo.
    Retorna:    pd.DataFrame: DataFrame contendo apenas as colunas selecionadas.
    """
    return df[feature_columns + [target_column]].copy()

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

sklearn.set_config(transform_output="pandas")
def scale_features(X_train, X_test, columns):
    """
    Padroniza colunas específicas de treino e teste usando StandardScaler,
    mantendo as demais colunas intactas.
    Argumentos: X_train: DataFrame de treino
                X_test: DataFrame de teste
                columns: Lista com os nomes das colunas a serem padronizadas
    
    Retorna:    X_train_scaled, X_test_scaled (ambos como DataFrames do Pandas)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), columns)
        ],
        remainder='passthrough'
    )
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    X_train_scaled.columns = X_train_scaled.columns.str.replace(r'^(num__|remainder__)', '', regex=True)
    X_test_scaled.columns = X_test_scaled.columns.str.replace(r'^(num__|remainder__)', '', regex=True)
    return X_train_scaled, X_test_scaled

def scale_full_features(X_full, columns):
    """Padroniza colunas do conjunto completo para treino e retorna os dados e o preprocessor ajustado."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), columns)
        ],
        remainder='passthrough'
    )
    X_full_scaled = preprocessor.fit_transform(X_full)
    X_full_scaled.columns = X_full_scaled.columns.str.replace(r'^(num__|remainder__)', '', regex=True)
    return X_full_scaled, preprocessor