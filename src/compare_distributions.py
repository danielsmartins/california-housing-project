import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from src.loader import load_data
from src.preprocessing import remove_outliers

def plot_before_after_grid():
    # Carregar dados
    df_raw = load_data()
    df_clean = remove_outliers(df_raw)
    
    features = ["AveRooms", "AveBedrms", "Population", "AveOccup"]
    
    save_dir = "outputs/figures"
    os.makedirs(save_dir, exist_ok=True)
    

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 16))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    fig.suptitle("Comparação de Distribuição: Antes vs. Depois da Limpeza (Percentil 99)", fontsize=16, y=0.92)
    
    for i, col in enumerate(features):
        # dados origniais
        sns.histplot(df_raw[col], bins=50, ax=axes[i, 0], color='skyblue', edgecolor='black', alpha=0.7)
        axes[i, 0].set_title(f"{col} - Original (Com Outliers)", fontsize=10, fontweight='bold')
        axes[i, 0].set_ylabel("Frequência")
        
        max_val = df_raw[col].max()
        axes[i, 0].annotate(f'Max: {max_val:.1f}', xy=(0.95, 0.8), xycoords='axes fraction', 
                            ha='right', fontsize=9, color='red')

        # dados limpos
        sns.histplot(df_clean[col], bins=50, ax=axes[i, 1], color='purple', edgecolor='black', alpha=0.7)
        axes[i, 1].set_title(f"{col} - Limpo (Sem 1% Topo)", fontsize=10, fontweight='bold')
        axes[i, 1].set_ylabel("") 
        

        new_max = df_clean[col].max()
        axes[i, 1].annotate(f'Novo Max: {new_max:.1f}', xy=(0.95, 0.8), xycoords='axes fraction', 
                            ha='right', fontsize=9, color='green')


    save_path = f"{save_dir}/distribution_comparison_grid.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Grid comparativo salvo em: {save_path}")

if __name__ == "__main__":
    plot_before_after_grid()