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