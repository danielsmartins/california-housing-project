import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers utilizando o limiar do percentil 99.
    """
    features_to_clean = ["AveRooms", "AveBedrms", "Population", "AveOccup"]
    df_clean = df.copy()
    
    initial_len = len(df_clean)
    
    print("remoção de outliers (Limiar 99%)")
    
    for col in features_to_clean:
        # Calcula o valor onde estão os 99% dos dados
        threshold = df_clean[col].quantile(0.99)
        
        # Filtra mantendo apenas quem é menor que o limite
        df_clean = df_clean[df_clean[col] < threshold]
    
    removed = initial_len - len(df_clean)
    print(f"   - Linhas removidas: {removed} (de {initial_len} para {len(df_clean)})")
    
    return df_clean


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    
    # feature engineering (Baseado em Géron, 2019)
    # A variável "BedroomsPerRoom" é citada pelo autor como tendo 
    # alta correlação com o valor da casa (geralmente casas com menos quartos 
    # por cômodo são mais caras/luxuosas ou têm layouts diferentes).
    df_feat["BedroomsPerRoom"] = df_feat["AveBedrms"] / df_feat["AveRooms"]
    
    # Transformações Logarítmicas e Substituição 
    # Criamos o Log e removemos original para evitar redundância/multicolinearidade
    
    # Renda
    df_feat["Log_MedInc"] = np.log1p(df_feat["MedInc"])
    df_feat.drop("MedInc", axis=1, inplace=True) # remove original
    
    # População
    df_feat["Log_Population"] = np.log1p(df_feat["Population"])
    df_feat.drop("Population", axis=1, inplace=True) # remove original
    
    return df_feat

def preprocess_data(df: pd.DataFrame):

    df = remove_outliers(df)
    
    print("Iniciando Pré-processamento (Split: Treino/Val/Teste) ")
    
    # Aplica a engenharia de features 
    df = feature_engineering(df)
    
    #  Separação: 
    # X (Features) = Tudo que usamos para prever
    # y (Target) = O que queremos prever (Preço da casa)
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    
    # Divisão dos dados (Splitting)
    # Primeiro corte: Tira 15% para o Teste Final
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    # Segundo corte: Do que sobrou (85%), tiramos uma parte para Validação.
    val_size_relative = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_relative, random_state=42
    )
    
    print(f" Dimensões: {df.shape}")
    print(f"🔹 Treino: {X_train.shape[0]} amostras")
    print(f"🔸 Validação: {X_val.shape[0]} amostras")
    print(f"🔻 Teste: {X_test.shape[0]} amostras")
    
    # Scaling (Normalização)
    scaler = StandardScaler()
    
    # Fit
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform na validação e teste usando a régua do treino
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertendo y para numpy array
    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()
    
    return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test), scaler, X.columns