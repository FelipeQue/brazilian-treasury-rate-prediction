import ssl
from io import StringIO
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

from src.config import RAW_FILE, START_DATE, END_DATE

def load_raw_data() -> pd.DataFrame:
    """Lê o CSV bruto em data/raw/ e devolve um DataFrame."""
    return pd.read_csv(RAW_FILE)


def _read_json_url(url: str) -> pd.DataFrame:
    """Lê JSON remoto com fallback para ambientes com cadeia de certificados local inconsistente."""
    try:
        with urlopen(url) as response:
            payload = response.read().decode("utf-8")
    except URLError:
        context = ssl._create_unverified_context()
        with urlopen(url, context=context) as response:
            payload = response.read().decode("utf-8")
    return pd.read_json(StringIO(payload))

def get_selic_sgs(start_date: str = START_DATE, end_date: str = END_DATE) -> pd.DataFrame:
    """Busca a série histórica da Selic (SGS 11) na API do Banco Central."""
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados?formato=json&dataInicial={start_date}&dataFinal={end_date}"
    df = _read_json_url(url)
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
    df.rename(columns={'data': 'DATA', 'valor': 'SELIC'}, inplace=True)
    return df

def get_dollar_sgs(start_date: str = START_DATE, end_date: str = END_DATE) -> pd.DataFrame:
    """Busca a série histórica do Dólar Comercial (SGS 1) na API do Banco Central."""
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados?formato=json&dataInicial={start_date}&dataFinal={end_date}"
    df = _read_json_url(url)
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
    df.rename(columns={'data': 'DATA', 'valor': 'USD_VALOR'}, inplace=True)
    return df