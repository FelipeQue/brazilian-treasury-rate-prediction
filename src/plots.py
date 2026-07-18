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

def plot_scatter(df: pd.DataFrame, x: str, y: str, title: str, filename: str) -> None:
    """Dispersão entre duas variáveis numéricas."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        ax=ax,
        color=GRAPH_CONFIG["primary_color"],
        alpha=0.6,
        edgecolor="white",
        linewidth=0.5)
    _style_axes(ax, title, x, y, filename)  

def plot_correlation_heatmap(df: pd.DataFrame, title: str, filename: str) -> None:
    """Mapa de calor da matriz de correlação entre variáveis numéricas."""
    corr = df.select_dtypes(include=np.number).corr(method="pearson")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        fmt=".2f",
        annot=False,
        cmap=GRAPH_CONFIG["heatmap_cmap"],
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="white",
        ax=ax)
    _style_axes(ax, title, "Variáveis", "Variáveis", filename)

def plot_boxplots(df: pd.DataFrame, cols: list[str], title: str, filename: str) -> None:
    """Gera boxplots para múltiplas colunas lado a lado"""
    fig, axes = plt.subplots(1, len(cols), figsize=(3.5 * len(cols), 5))
    if len(cols) == 1:
        axes = [axes]        
    for ax, col in zip(axes, cols):
        sns.boxplot(
            data=df,
            y=col,
            ax=ax,
            color=GRAPH_CONFIG["primary_color"],
            linewidth=1.5,
            fliersize=4,
            flierprops={"markerfacecolor": GRAPH_CONFIG["dark_color"], "markeredgecolor": "white"}
        )
        
        ax.set_title(col, fontsize=14, fontweight="bold", pad=10, color=GRAPH_CONFIG["title_color"])
        ax.set_ylabel(col, fontsize=11, color=GRAPH_CONFIG["text_color"])
        ax.set_xlabel("")
        
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02, color=GRAPH_CONFIG["title_color"])
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    _savefig(filename)
    plt.show()

def plot_time_series(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    title: str,
    filename: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Gera um gráfico de série temporal de uma variável numérica ao longo do tempo."""
    df_plot = df.copy()
    df_plot[date_column] = pd.to_datetime(df_plot[date_column])
    x_label = xlabel or date_column
    y_label = ylabel or value_column
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(
        data=df_plot,
        x=date_column,
        y=value_column,
        ax=ax,
        color=GRAPH_CONFIG["primary_color"],
        linewidth=2.5
    )
    _style_axes(ax, title, x_label, y_label, filename)
    plt.close(fig)

def plot_observed_vs_predicted(
    y_true: np.ndarray | pd.Series, 
    y_pred: np.ndarray | pd.Series, 
    title: str, 
    filename: str,
    xlabel: str = "Valor Real",
    ylabel: str = "Valor Previsto"
) -> None:
    """Plota gráfico dos valores reais x valores previstos."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    ax.scatter(
        y_true, 
        y_pred, 
        alpha=0.4, 
        s=20, 
        color=GRAPH_CONFIG["primary_color"], 
        edgecolor="white", 
        linewidth=0.3
    )
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color=GRAPH_CONFIG["dark_color"], linewidth=2.0, label="Ajuste perfeito")
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    _style_axes(ax, title, xlabel, ylabel, filename)


def plot_residuals(
    y_true: np.ndarray | pd.Series, 
    y_pred: np.ndarray | pd.Series, 
    title: str, 
    filename: str,
    xlabel: str = "Valor Previsto",
    ylabel: str = "Resíduo (Real - Previsto)"
) -> pd.Series:
    """Resíduos (real - previsto) versus valores previstos. Devolve os resíduos calculados."""
    residuals = pd.Series(np.asarray(y_true) - np.asarray(y_pred))
    
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        y_pred, 
        residuals, 
        alpha=0.5, 
        color=GRAPH_CONFIG["primary_color"], 
        edgecolor="white", 
        linewidth=0.3
    )
    
    ax.axhline(0, color=GRAPH_CONFIG["dark_color"], linestyle="--", linewidth=1.8)
    _style_axes(ax, title, xlabel, ylabel, filename)
    return residuals