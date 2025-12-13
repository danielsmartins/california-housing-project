import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_artifacts():
    print("Carregando o modelo e dados de teste...")
    processed_path = "data/processed"
    
    # Carregamento o modelo 
    model = joblib.load(f"{processed_path}/best_model_mlp.pkl")
    
    # Carregamos o conjunto de teste (Blind Test)
    X_test, y_test = joblib.load(f"{processed_path}/test_data.pkl")
    
    return model, X_test, y_test

def plot_predicted_vs_actual(y_true, y_pred, r2):
    """
    Gera o gráfico equivalente à 'Acurácia' para Regressão.
    Mostra o quão perto os pontos estão da linha ideal.
    """
    plt.figure(figsize=(10, 8))
    
    # Scatter plot com transparência para ver onde os dados se aglomeram
    plt.scatter(y_true, y_pred, alpha=0.4, color='#1f77b4', edgecolor='k', s=20, label='Predições')
    
    # Linha Perfeita (y = x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Ideal (Erro Zero)')
    
    plt.title(f"Predito vs. Real (Conjunto de Teste)\nR² = {r2:.4f} (Quanto mais próximo de 1.0, melhor)", fontsize=14)
    plt.xlabel("Valor Real da Casa ($100k)", fontsize=12)
    plt.ylabel("Valor Predito pelo Modelo ($100k)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "outputs/figures/predicted_vs_actual.png"
    plt.savefig(save_path, dpi=300)
    print(f"📊 Gráfico Predito x Real salvo em: {save_path}")

def plot_residuals(y_true, y_pred):
    """
    Histograma dos erros. Mostra se o modelo tende a chutar para cima ou para baixo.
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, color='purple', alpha=0.7, edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', lw=2)
    plt.title("Distribuição dos Erros (Resíduos)", fontsize=14)
    plt.xlabel("Erro ($100k) - (Real - Predito)")
    plt.ylabel("Frequência")
    
    save_path = "outputs/figures/residuals_hist.png"
    plt.savefig(save_path, dpi=300)
    print(f"📊 Histograma de Resíduos salvo em: {save_path}")

def evaluate_model():
    model, X_test, y_test = load_artifacts()
    
    print("Realizando inferência no conjunto de teste")
    y_pred = model.predict(X_test)
    
    # cáculo de métricas
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print(f"{'RESULTADOS FINAIS (TEST SET)':^40}")
    print("="*40)
    print(f" RMSE (Erro Médio Quadrático): {rmse:.4f} ($ {rmse*100:.0f}k)")
    print(f" MAE  (Erro Médio Absoluto):   {mae:.4f} ($ {mae*100:.0f}k)")
    print(f" R²   (Score de Variância):    {r2:.4f} (0 a 1)")
    print("-" * 40)
    
    # Gerar gráficos para o relatório
    plot_predicted_vs_actual(y_test, y_pred, r2)
    plot_residuals(y_test, y_pred)

if __name__ == "__main__":
    evaluate_model()