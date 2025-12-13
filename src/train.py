import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import os

# Ignorar avisos de convergência (esperado pois controlamos o loop manualmente)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def load_processed_data():
    """Carrega os dados preparados anteriormente."""
    print("🔄 Carregando dados processados...")
    processed_path = "data/processed"
    
    # Carregando tuplas (X, y)
    X_train, y_train = joblib.load(f"{processed_path}/train_data.pkl")
    X_val, y_val     = joblib.load(f"{processed_path}/val_data.pkl")
    
    # Carregando feature names apenas para log
    feature_names = joblib.load(f"{processed_path}/feature_names.pkl")
    
    print(f"✅ Dados carregados. Features: {len(feature_names)}")
    return X_train, y_train, X_val, y_val

def train_model():
    X_train, y_train, X_val, y_val = load_processed_data()
    
    # config do MLP 
    # hidden_layer_sizes=(64, 32): Duas camadas ocultas.
    #   - 1ª com 64 neurônios 
    #   - 2ª com 32 neurônios 
    # activation='relu': Padrão moderno para Deep Learning.
    # solver='adam': Otimizador 
    # warm_start=True:  Permite treinar época por época sem resetar os pesos.
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.05,           # Regularização para evitar overfitting
        learning_rate_init=0.001,
        max_iter=1,            # Treina 1 época por vez no loop 
        warm_start=True,       # Mantém a memória entre os loops
        random_state=42,
        verbose=False
    )
    
    # Listas para guardar o histórico (para o gráfico do relatório)
    train_loss_history = []
    val_loss_history = []
    
    epochs = 100  # Número total de épocas
    print(f"\n Iniciando treinamento por {epochs} épocas...")
    print(f"{'Época':^10} | {'Train RMSE':^12} | {'Val RMSE':^12}")
    print("-" * 40)
    
    best_val_loss = float('inf')
    no_improvement_count = 0
    patience = 15  # Early Stopping: para se não melhorar após 15 épocas
    
    for epoch in range(1, epochs + 1):
        # Treina por 1 época
        model.fit(X_train, y_train)
        
        # Avalia desempenho atual
        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)
        
        # Calculando RMSE (Root Mean Squared Error)
        train_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
        val_rmse = np.sqrt(mean_squared_error(y_val, pred_val))
        
        # Salvando histórico
        train_loss_history.append(train_rmse)
        val_loss_history.append(val_rmse)
        
        # Log a cada 10 épocas
        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:^10} | {train_rmse:^12.4f} | {val_rmse:^12.4f}")
            
        # Lógica de Early Stopping Manual 
        # Se o erro na validação for o menor até agora, salvamos esse modelo
        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            no_improvement_count = 0
            joblib.dump(model, "data/processed/best_model_mlp.pkl")
        else:
            no_improvement_count += 1
            
        if no_improvement_count >= patience:
            print(f"\nEarly Stopping ativado na época {epoch}. Sem melhoria por {patience} épocas.")
            break
            
    print("\n Treinamento finalizado")
    
    # Plotando o gráfico de Loss
    plot_loss_curve(train_loss_history, val_loss_history)

def plot_loss_curve(train_loss, val_loss):
    """Gera o gráfico comparativo de erro Treino vs Validação."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Treino (RMSE)')
    plt.plot(val_loss, label='Validação (RMSE)', linestyle='--')
    
    plt.title('Curva de Aprendizado (Loss Curve)')
    plt.xlabel('Épocas')
    plt.ylabel('Erro (RMSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "outputs/figures/loss_curve.png"
    plt.savefig(save_path, dpi=300)
    print(f" Gráfico de Loss salvo em: {save_path}")

if __name__ == "__main__":
    train_model()