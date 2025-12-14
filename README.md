# California Housing Price Predictor (MLP)

Este projeto implementa um estimador de preços de imóveis utilizando uma Rede Neural Artificial (MLPRegressor) no dataset California Housing.

## 🛠️ Instalação e Configuração

**Pré-requisitos:** Python 3.8+

**Clone o repositório:**

```bash
git clone <SEU_LINK_DO_GITHUB_AQUI>
cd california_housing_project
```
## Crie um ambiente virtual (Recomendado):

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

## Instale as dependências:

```bash
pip install -r requirements.txt
```

# 🚀 Como Rodar o Projeto
Como os modelos treinados não foram incluídos no repositório, siga a ordem abaixo para gerar os dados, treinar a rede e avaliar os resultados.

## Passo 1: Preparação dos Dados
Este script baixa o dataset original, realiza a limpeza (remoção de outliers), aplica a engenharia de features (Logs e Razões) e salva os arquivos processados na pasta data/processed.

```bash
python main_analysis.py
```
## Passo 2: Treinamento do Modelo
Este script carrega os dados processados e treina a Rede Neural (MLP).
Configuração: 2 Camadas Ocultas (64, 32), Otimizador Adam, Regularização L2.
Saída: O modelo treinado será salvo em data/processed/best_model_mlp.pkl.

```bash
python src/train.py
```
## Passo 3: Avaliação e Resultados
Gera as métricas finais (R², RMSE, MAE) no conjunto de teste e cria os gráficos de performance (Predito vs. Real e Resíduos).

```bash
python src/evaluate.py
```
## 📊 Gerar Visualizações Extras
Para gerar os gráficos utilizados na análise exploratória e documentação (arquitetura da rede e distribuições):

```bash
# Gerar diagrama da arquitetura da rede neural
python -m src.visualize_architecture

# Comparar distribuições (Antes vs. Depois da limpeza)
python -m src.compare_distributions
As imagens serão salvas na pasta outputs/figures.
````

📁 Estrutura do Projeto
```text
├── data/                  # Dados brutos e processados (.pkl)
├── outputs/figures/       # Gráficos gerados
├── src/
│   ├── loader.py          # Carregamento de dados
│   ├── preprocessing.py   # Limpeza e Feature Engineering
│   ├── train.py           # Loop de treinamento
│   ├── evaluate.py        # Cálculo de métricas e gráficos finais
│   └── ...
├── main_analysis.py       # Orchestrador de preparação de dados
└── requirements.txt       # Bibliotecas necessárias
```
