"""
Caminhos e parâmetros centrais do projeto, usados pelo notebook e pelos
módulos de src/.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = ROOT_DIR / "data" / "raw" / "leiloes_tesouro_nacional.csv"