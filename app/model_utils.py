import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.decomposition import PCA
import os

# Modelos do Prof. Miro
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA


def load_data(file_path):
    """Carrega os dados do CSV."""
    df = pd.read_csv(file_path)

    # --- CORREÇÃO (PATCH 3) ---
    if 'time' not in df.columns:
        return df
    # --- FIM DA CORREÇÃO ---

    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])
    return df

def preprocess_data(df, test_size=0.2):
    """
    Prepara os dados para os modelos.
    NOTA: shuffle=False é CRÍTICO para Séries Temporais.
    """
    X = df.drop('time', axis=1)
    y = df['time']
    
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False 
    )

    # --- Pipeline 1 (Base PI) -> MinMaxScaler ---
    scaler_minmax = MinMaxScaler()
    X_train_scaled_minmax = scaler_minmax.fit_transform(X_train_df)
    X_test_scaled_minmax = scaler_minmax.transform(X_test_df)

    # --- Pipelines 2 (PCA), 3 (Ridge), 4 (Lasso) -> StandardScaler ---
    scaler_std = StandardScaler()
    X_train_scaled_std = scaler_std.fit_transform(X_train_df)
    X_test_scaled_std = scaler_std.transform(X_test_df)
    
    # Pipeline 2 (PCA)
    pca = PCA(n_components=0.95) 
    X_train_pca = pca.fit_transform(X_train_scaled_std)
    X_test_pca = pca.transform(X_test_scaled_std)
    
    # Para Validação Cruzada
    X_full_scaled_minmax = scaler_minmax.transform(X) 
    X_full_scaled_std = scaler_std.transform(X)       
    X_full_pca = pca.transform(X_full_scaled_std)       

    return (
        (X_train_scaled_minmax, X_test_scaled_minmax, scaler_minmax), # Pip 1
        (X_train_pca, X_test_pca, scaler_std, pca),                 # Pip 2
        (X_train_scaled_std, X_test_scaled_std, scaler_std),        # Pips 3 e 4
        (y_train, y_test),                                          # Pips 5 e 6
        (X_full_scaled_minmax, X_full_pca, X_full_scaled_std, y)    # Para Validação
    )

def get_expected_performance_sklearn(X_full_scaled_minmax, X_full_pca, X_full_scaled_std, y):
    """
    Usa TimeSeriesSplit (Prof. Miro) para avaliar os 4 modelos sklearn.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    
    models = {
        "linear_minmax": LinearRegression(),
        "linear_pca": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=1.0)
    }
    
    data_X = {
        "linear_minmax": X_full_scaled_minmax,
        "linear_pca": X_full_pca,
        "ridge": X_full_scaled_std,
        "lasso": X_full_scaled_std
    }
    
    results = { "linear_minmax": [], "linear_pca": [], "ridge": [], "lasso": [] }

    for model_name in models:
        for train_index, test_index in tscv.split(data_X[model_name]):
            X_train, X_test = data_X[model_name][train_index], data_X[model_name][test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model = models[model_name]
            model.fit(X_train, y_train)
            try:
                score = model.score(X_test, y_test) # R²
                results[model_name].append(score)
            except:
                 results[model_name].append(np.nan) 

    return (
        np.nanmean(results["linear_minmax"]), 
        np.nanmean(results["linear_pca"]), 
        np.nanmean(results["ridge"]), 
        np.nanmean(results["lasso"])
    )

def train_sklearn_model(X_train, y_train, model_type='linear'):
    """Treina os modelos sklearn (Linear, Ridge, Lasso)."""
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0) 
    elif model_type == 'lasso':
        model = Lasso(alpha=1.0) 
    model.fit(X_train, y_train)
    return model

def train_holtwinters_model(y_train, y_test):
    """
    Treina o modelo Holt-Winters (Prof. Miro)
    """
    hw_model = ExponentialSmoothing(
        y_train, 
        trend='add', 
        seasonal='add', 
        seasonal_periods=5 
    ).fit()
    
    y_pred_hw = hw_model.forecast(steps=len(y_test))
    expected_r2 = r2_score(y_test, y_pred_hw)
    return hw_model, expected_r2

def train_arima_model(y_train, y_test):
    """
    Treina o modelo ARIMA (Prof. Miro)
    """
    arima_model = ARIMA(y_train, order=(1, 1, 1)).fit()
    y_pred_arima = arima_model.forecast(steps=len(y_test))
    expected_r2 = r2_score(y_test, y_pred_arima)
    return arima_model, expected_r2

# --- REMOVIDO: Função train_lstm_model ---

def preprocess_unseen_data(X_unseen, scaler_minmax, scaler_std, pca):
    """Aplica as transformações aprendidas (Scalers e PCA) aos novos dados."""
    
    X_unseen_scaled_minmax = scaler_minmax.transform(X_unseen)
    X_unseen_scaled_std = scaler_std.transform(X_unseen)
    X_unseen_pca = pca.transform(X_unseen_scaled_std)
    
    return X_unseen_scaled_minmax, X_unseen_pca, X_unseen_scaled_std