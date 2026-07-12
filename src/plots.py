import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.config import IMAGES_DIR, GRAPH_CONFIG

def _savefig(filename: str) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(IMAGES_DIR / filename, bbox_inches="tight", dpi=150)

def _style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str, filename: str) -> None:
    """Função utilitária interna para aplicar a padronização visual e salvar o gráfico."""
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20, color=GRAPH_CONFIG["title_color"])
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10, color=GRAPH_CONFIG["text_color"])
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10, color=GRAPH_CONFIG["text_color"])
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    _savefig(filename)
    plt.show()

def plot_histogram(df: pd.DataFrame, column: str, title: str, filename: str) -> None:
    """Histograma com curva de densidade (KDE) de uma variável numérica."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(
        data=df,
        x=column,
        bins=60,
        kde=True,
        ax=ax, 
        color=GRAPH_CONFIG["primary_color"],
        edgecolor="white",
        alpha=0.8,
        linewidth=0.5,
        line_kws={"color": GRAPH_CONFIG["dark_color"], "linewidth": 2.5})
    _style_axes(ax, title, column, "Frequência", filename)

def plot_histogram_by_category(df: pd.DataFrame, column: str, hue: str, title: str, filename: str) -> None:
    """Histograma segmentado por categoria sem curva de densidade."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(
        data=df,
        x=column,
        hue=hue,
        bins=60,
        kde=False,
        ax=ax,
        palette=GRAPH_CONFIG["heatmap_cmap"],
        edgecolor="white",
        alpha=0.8,
        linewidth=0.5)
    _style_axes(ax, title, column, "Frequência", filename)