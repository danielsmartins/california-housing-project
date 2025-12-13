import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_neural_network():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Configurações das camadas
    layer_sizes = [9, 64, 32, 1]  # Input(9), Hidden1(64), Hidden2(32), Output(1)
    layer_names = ['Input Layer\n(9 Features)', 'Hidden Layer 1\n(64 Neurônios + ReLU)', 
                   'Hidden Layer 2\n(32 Neurônios + ReLU)', 'Output Layer\n(Preço Estimado)']
    
    # Posições X das camadas
    x_positions = [0.1, 0.35, 0.6, 0.85]
    
    # Desenhar conexões (simplificadas para não poluir)
    for i in range(len(layer_sizes) - 1):
        # Desenha uma seta grande representando o fluxo denso
        ax.arrow(x_positions[i] + 0.05, 0.5, 
                 x_positions[i+1] - x_positions[i] - 0.12, 0, 
                 head_width=0.05, head_length=0.02, fc='gray', ec='gray', alpha=0.5)
        
        # Texto da transformação
        if i < 2:
            ax.text((x_positions[i] + x_positions[i+1])/2, 0.55, "Dense + ReLU", 
                    ha='center', fontsize=9, color='darkblue')
        else:
            ax.text((x_positions[i] + x_positions[i+1])/2, 0.55, "Linear", 
                    ha='center', fontsize=9, color='darkblue')

    # Desenhar as "Caixas" representando as camadas
    colors = ['#ADD8E6', '#90EE90', '#90EE90', '#FFB6C1'] # Azul, Verde, Verde, Vermelho
    
    for i, (size, name, x) in enumerate(zip(layer_sizes, layer_names, x_positions)):
        # Desenha Círculos simbólicos
        num_circles = min(size, 8) # Não desenhar 64 bolinhas, desenhar simbólico
        y_start = 0.3
        y_end = 0.7
        step = (y_end - y_start) / (num_circles - 1) if num_circles > 1 else 0
        
        # Caixa ao redor
        rect = patches.FancyBboxPatch((x - 0.08, 0.2), 0.16, 0.6, 
                                      boxstyle="round,pad=0.02", 
                                      linewidth=1, edgecolor='black', facecolor='none', alpha=0.2)
        ax.add_patch(rect)
        
        for j in range(num_circles):
            y = y_start + j * step if num_circles > 1 else 0.5
            circle = plt.Circle((x, y), 0.02, color=colors[i], ec='black')
            ax.add_patch(circle)
            
        # Rótulos
        ax.text(x, 0.85, name, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(x, 0.15, f"Shape: ({size})", ha='center', va='center', fontsize=9, style='italic')

    plt.title("Arquitetura do Modelo MLP Implementado", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig("outputs/figures/model_architecture.png", dpi=300)
    print("📊 Diagrama da arquitetura salvo em outputs/figures/model_architecture.png")

if __name__ == "__main__":
    draw_neural_network()