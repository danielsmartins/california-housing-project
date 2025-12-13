import pandas as pd
from sklearn.datasets import fetch_california_housing

def load_data() -> pd.DataFrame:
    """
    Carrega o dataset California Housing do Scikit-Learn.
    Retorna um DataFrame com as features e o target ('MedHouseVal').
    """
    print("Carregando dataset California Housing")
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    print(f" Dados carregado Shape: {df.shape}")
    return df