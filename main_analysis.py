from src.loader import load_data
from src.analysis import run_eda
from src.preprocessing import preprocess_data
import joblib
import os

def main():
    # Carregar Dados
    df = load_data()
    
    # 2. Análise Exploratória (Gera as figuras para o relatório)
    run_eda(df)
    
    # Pré-processamento
    train_set, val_set, test_set, scaler, feature_names = preprocess_data(df)
    
    # Desempacotando os conjuntos
    X_train, y_train = train_set
    X_val, y_val = val_set
    X_test, y_test = test_set
    
    # 4. Salvar os dados processados
    output_path = "data/processed"
    os.makedirs(output_path, exist_ok=True)
    
    print(f"\n Salvando datasets em '{output_path}'...")
    joblib.dump((X_train, y_train), f"{output_path}/train_data.pkl")
    joblib.dump((X_val, y_val),     f"{output_path}/val_data.pkl")
    joblib.dump((X_test, y_test),   f"{output_path}/test_data.pkl")
    
    # Salvando artefatos auxiliares
    joblib.dump(scaler, f"{output_path}/scaler.pkl")
    joblib.dump(feature_names, f"{output_path}/feature_names.pkl")
    
    print("concluído")


if __name__ == "__main__":
    main()