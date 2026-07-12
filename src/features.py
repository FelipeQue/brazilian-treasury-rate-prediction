import pandas as pd

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