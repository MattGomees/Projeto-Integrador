# app/model_utils.py

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# --- Importações do Scikit-learn ---
# O Pipeline é o objeto que "amarra" os passos
from sklearn.pipeline import Pipeline
# O Scaler para normalização Min-Max
from sklearn.preprocessing import MinMaxScaler
# O modelo de Regressão Linear
from sklearn.linear_model import LinearRegression
# O validador cruzado para Séries Temporais
from sklearn.model_selection import TimeSeriesSplit
# As métricas de avaliação de regressão
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# --- Constantes de Caminho ---
# Define o caminho absoluto para a pasta 'artifacts'
# __file__ é o próprio arquivo (model_utils.py)
# os.path.dirname(__file__) é a pasta 'app/'
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
# Garante que a pasta 'artifacts' exista
os.makedirs(ARTIFACT_DIR, exist_ok=True)
# Define o caminho completo para o arquivo do modelo salvo
ARTIFACT_MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.joblib")


def build_pipeline() -> Pipeline:
    """Cria o pipeline de ML (Scaler + Modelo)"""
    pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("linreg", LinearRegression())
    ])
    return pipeline


def _select_features(df: pd.DataFrame, target_col: str, feature_cols: Optional[List[str]]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Helper para separar X (features) e y (target)"""
    
    # Se uma lista de features não foi dada, infere
    if feature_cols is None:
        # Usa todas as colunas, exceto a coluna target
        feature_cols = [col for col in df.columns if col != target_col]
        
    X = df[feature_cols]
    
    # O target pode não existir em dados de aplicação (teste sem rótulo)
    y = None
    if target_col in df.columns:
        y = df[target_col]
        
    return X, y, feature_cols


def timeseries_expected_performance(
    df_train: pd.DataFrame, 
    target_col: str, 
    feature_cols: Optional[List[str]], 
    n_splits: int = 5
) -> Dict[str, float]:
    """
    Calcula a expectativa de desempenho usando TimeSeriesSplit.
    Este é um requisito do PI para avaliação "adequada".
    """
    print(f"Calculando expectativa de desempenho com TimeSeriesSplit (n_splits={n_splits})...")
    
    # 1. Separar X e y
    X, y, _ = _select_features(df_train, target_col, feature_cols)
    if y is None:
        raise ValueError("Dados de treino não podem estar sem a coluna target.")

    # 2. Configurar o validador temporal
    # test_size=None faz ele usar cada fold seguinte como teste
    # ex: Fold 1: Treino[0:100], Teste[100:150]
    #     Fold 2: Treino[0:150], Teste[150:200]
    tscv = TimeSeriesSplit(n_splits=n_splits)

    metrics_list = []

    # 3. Iterar sobre os folds
    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # 4. Criar e treinar um pipeline NOVO para cada fold
        # Isso é crucial para evitar data leakage
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)
        
        # 5. Fazer predições e calcular métricas
        y_pred = pipeline.predict(X_val)
        
        metrics = evaluate_predictions(y_val, y_pred)
        metrics['fold'] = fold + 1
        metrics_list.append(metrics)
        
    # 6. Calcular a média das métricas de todos os folds
    df_metrics = pd.DataFrame(metrics_list)
    
    # Retorna a média como a "expectativa"
    expected_metrics = {
        "exp_mae": df_metrics['mae'].mean(),
        "exp_rmse": df_metrics['rmse'].mean(),
        "exp_r2": df_metrics['r2'].mean(),
        "exp_mape": df_metrics['mape'].mean()
    }
    
    print(f"Expectativa de desempenho calculada: {expected_metrics}")
    return expected_metrics


def train_full_model(
    df_train: pd.DataFrame, 
    target_col: str, 
    feature_cols: Optional[List[str]]
) -> None:
    """
    Treina o modelo final com TODOS os dados de treino e salva 
    o artefato (model.joblib) em disco.
    """
    print("Treinando modelo final com todos os dados de treino...")
    
    # 1. Separar X e y
    X, y, final_feature_cols = _select_features(df_train, target_col, feature_cols)
    if y is None:
        raise ValueError("Dados de treino não podem estar sem a coluna target.")

    # 2. Construir e treinar o pipeline
    pipeline = build_pipeline()
    pipeline.fit(X, y)

    # 3. Criar o dicionário de artefatos
    #    Salvamos o pipeline E a lista de features
    #    Isso é vital para garantir a ordem na predição!
    artifacts = {
        "pipeline": pipeline,
        "features": final_feature_cols,
        "target": target_col
    }

    # 4. Salvar em disco
    joblib.dump(artifacts, ARTIFACT_MODEL_PATH)
    print(f"Modelo salvo em {ARTIFACT_MODEL_PATH}")


def load_model_artifacts() -> Dict:
    """
    Carrega os artefatos (pipeline, features) do disco.
    """
    if not os.path.exists(ARTIFACT_MODEL_PATH):
        raise FileNotFoundError(
            "Artefato do modelo (model.joblib) não encontrado. "
            "Por favor, treine o modelo primeiro (endpoint /train)."
        )
        
    print(f"Carregando modelo de {ARTIFACT_MODEL_PATH}...")
    artifacts = joblib.load(ARTIFACT_MODEL_PATH)
    return artifacts


def predict_dataframe(df_test: pd.DataFrame, artifacts: Dict) -> pd.Series:
    """
    Gera previsões para um novo DataFrame usando o pipeline carregado.
    """
    print("Gerando previsões...")
    
    # 1. Extrair o pipeline e a ordem de features salvas
    pipeline = artifacts['pipeline']
    feature_cols = artifacts['features']

    # 2. Garantir que o df_test tenha as colunas necessárias
    try:
        X_test = df_test[feature_cols]
    except KeyError as e:
        raise ValueError(
            f"Coluna de feature ausente nos dados de teste: {e}. "
            f"Features esperadas: {feature_cols}"
        )

    # 3. Gerar predições
    # O pipeline.predict() aplica o .transform() do scaler
    # e depois o .predict() do modelo, tudo automaticamente.
    y_pred = pipeline.predict(X_test)
    
    return pd.Series(y_pred, name="predictions")


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Calcula o dicionário de métricas de regressão."""
    
    # Evitar divisão por zero no MAPE se y_true tiver zeros
    # Substitui 0 por um valor muito pequeno (epsilon)
    y_true_safe = y_true.replace(0, np.finfo(float).eps)

    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true_safe, y_pred) * 100 # Em %
    }
    return metrics


def reset_artifacts() -> None:
    """
    Apaga os artefatos de modelo salvos (cumpre o requisito /reset).
    """
    if os.path.exists(ARTIFACT_MODEL_PATH):
        os.remove(ARTIFACT_MODEL_PATH)
        print(f"Artefato {ARTIFACT_MODEL_PATH} removido.")
    else:
        print("Nenhum artefato de modelo para remover.")