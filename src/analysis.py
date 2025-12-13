import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_correlation_matrix(df: pd.DataFrame, save_path: str):
    """Gera e salva a matriz de correlação (Heatmap)."""
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Matriz de Correlação - California Housing")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "correlation_matrix.png"), dpi=300)
    plt.close()
    print("📊 Matriz de correlação salva.")

def plot_distributions(df: pd.DataFrame, save_path: str):
    """Gera histogramas para entender a distribuição dos dados."""
    df.hist(bins=50, figsize=(20, 15))
    plt.suptitle("Distribuição das Features")
    plt.savefig(os.path.join(save_path, "distributions.png"), dpi=300)
    plt.close()
    print("📊 Histogramas de distribuição salvos.")

def plot_geospatial(df: pd.DataFrame, save_path: str):
    """
    Plota as casas pela latitude/longitude.
    Cor = Preço, Tamanho = População.
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(
        df['Longitude'], df['Latitude'], 
        alpha=0.4, 
        s=df['Population']/100, 
        label='População', 
        c=df['MedHouseVal'], 
        cmap='jet'
    )
    plt.colorbar(label='Valor Médio da Casa (MedHouseVal)')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Distribuição Geográfica de Preços e População")
    plt.legend()
    plt.savefig(os.path.join(save_path, "geospatial_plot.png"), dpi=300)
    plt.close()
    print("📊 Gráfico geoespacial salvo.")

def run_eda(df: pd.DataFrame):
    output_dir = "outputs/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Iniciando Análise Exploratória")
    plot_correlation_matrix(df, output_dir)
    plot_distributions(df, output_dir)
    plot_geospatial(df, output_dir)