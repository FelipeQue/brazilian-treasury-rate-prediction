"""
Caminhos e parâmetros centrais do projeto, usados pelo notebook e pelos
módulos de src/.
"""
from pathlib import Path
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = ROOT_DIR / "data" / "raw" / "leiloes_tesouro_nacional.csv"

START_DATE = "01/01/2015"
END_DATE = "31/12/2023"

IMAGES_DIR = ROOT_DIR / "outputs" / "images"

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "leiloes_tesouro_nacional_processed.csv"

FINAL_DIR = ROOT_DIR / "data" / "final"
FINAL_FILE = FINAL_DIR / "leiloes_tesouro_nacional_final.csv"

MODEL_DIR = ROOT_DIR / "models"

# Configurações de estilo do Seaborn e cores para os gráficos
sns.set_theme(style="whitegrid", rc={"grid.linestyle": "--", "grid.alpha": 0.5})
_palette = sns.color_palette("rocket", as_cmap=False)

GRAPH_CONFIG = {
    "title_color": "#1A1126",
    "text_color": "#333333",
    "primary_color": _palette[2],
    "dark_color": _palette[1],
    "heatmap_cmap": "rocket_r"
}