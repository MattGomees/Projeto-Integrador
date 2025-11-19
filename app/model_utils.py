import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.decomposition import PCA
import os
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA


def load_data(file_path):
    df = pd.read_csv(file_path)
    if 'time' not in df.columns:
        return df
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])
    return df

def preprocess_data(df, test_size=0.2):
    X = df.drop('time', axis=1)
    y = df['time']
    
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False 
    )

    scaler_minmax = MinMaxScaler()
    X_train_scaled_minmax = scaler_minmax.fit_transform(X_train_df)
    X_test_scaled_minmax = scaler_minmax.transform(X_test_df)

    scaler_std = StandardScaler()
    X_train_scaled_std = scaler_std.fit_transform(X_train_df)
    X_test_scaled_std = scaler_std.transform(X_test_df)
    
    pca = PCA(n_components=0.90) 
    X_train_pca = pca.fit_transform(X_train_scaled_std)
    X_test_pca = pca.transform(X_test_scaled_std)
    
    X_full_scaled_minmax = scaler_minmax.transform(X) 
    X_full_scaled_std = scaler_std.transform(X)       
    X_full_pca = pca.transform(X_full_scaled_std)       

    return (
        (X_train_scaled_minmax, X_test_scaled_minmax, scaler_minmax),
        (X_train_pca, X_test_pca, scaler_std, pca),
        (X_train_scaled_std, X_test_scaled_std, scaler_std),
        (y_train, y_test),
        (X_full_scaled_minmax, X_full_pca, X_full_scaled_std, y)
    )

def get_expected_performance_sklearn(X_full_scaled_minmax, X_full_pca, X_full_scaled_std, y):
    tscv = TimeSeriesSplit(n_splits=5)
    
    models = {
        "linear_minmax": LinearRegression(),
        "linear_pca": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.1)
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
                score = model.score(X_test, y_test)
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
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0) 
    elif model_type == 'lasso':
        model = Lasso(alpha=0.1) 
    model.fit(X_train, y_train)
    return model

def train_holtwinters_model(y_train, y_test):
    best_aic = float('inf')
    best_model = None
    
    configs = [
        {'trend': 'add', 'seasonal': None},
        {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 5},
        {'trend': None, 'seasonal': 'add', 'seasonal_periods': 5}
    ]
    
    for config in configs:
        try:
            model = ExponentialSmoothing(
                y_train, 
                trend=config.get('trend'), 
                seasonal=config.get('seasonal'), 
                seasonal_periods=config.get('seasonal_periods')
            ).fit()
            
            if model.aic < best_aic:
                best_aic = model.aic
                best_model = model
        except:
            continue
            
    if best_model is None:
        best_model = ExponentialSmoothing(y_train).fit()

    y_pred_hw = best_model.forecast(steps=len(y_test))
    expected_r2 = r2_score(y_test, y_pred_hw)
    return best_model, expected_r2

def train_arima_model(y_train, y_test):
    best_aic = float('inf')
    best_order = (1, 1, 1)
    best_model_fit = None
    
    p_values = [0, 1, 2, 4]
    d_values = [0, 1]
    q_values = [0, 1, 2]
    
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(y_train, order=(p, d, q))
                    model_fit = model.fit()
                    
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p, d, q)
                        best_model_fit = model_fit
                except:
                    continue
    
    if best_model_fit is None:
        best_model_fit = ARIMA(y_train, order=(1, 1, 1)).fit()

    y_pred_arima = best_model_fit.forecast(steps=len(y_test))
    expected_r2 = r2_score(y_test, y_pred_arima)
    
    return best_model_fit, expected_r2

def preprocess_unseen_data(X_unseen, scaler_minmax, scaler_std, pca):
    X_unseen_scaled_minmax = scaler_minmax.transform(X_unseen)
    X_unseen_scaled_std = scaler_std.transform(X_unseen)
    X_unseen_pca = pca.transform(X_unseen_scaled_std)
    return X_unseen_scaled_minmax, X_unseen_pca, X_unseen_scaled_std

def train_models(X_train_scaled, X_test_scaled, 
                 X_train_pca, X_test_pca,
                 X_train_scaled_std, X_test_scaled_std,
                 y_train, y_test):
    
    model_scaled = train_sklearn_model(X_train_scaled, y_train, model_type='linear')
    pred_scaled = model_scaled.predict(X_test_scaled)
    r2_scaled = r2_score(y_test, pred_scaled)

    model_pca = train_sklearn_model(X_train_pca, y_train, model_type='linear')
    pred_pca = model_pca.predict(X_test_pca)
    r2_pca = r2_score(y_test, pred_pca)

    model_ridge = train_sklearn_model(X_train_scaled_std, y_train, model_type='ridge')
    pred_ridge = model_ridge.predict(X_test_scaled_std)
    r2_ridge = r2_score(y_test, pred_ridge)

    model_lasso = train_sklearn_model(X_train_scaled_std, y_train, model_type='lasso')
    pred_lasso = model_lasso.predict(X_test_scaled_std)
    r2_lasso = r2_score(y_test, pred_lasso)

    model_hw, r2_hw = train_holtwinters_model(y_train, y_test)
    model_arima, r2_arima = train_arima_model(y_train, y_test)

    metrics = {
        "Linear_Scaled_R2": r2_scaled,
        "Linear_PCA_R2": r2_pca,
        "Ridge_R2": r2_ridge,
        "Lasso_R2": r2_lasso,
        "HoltWinters_R2": r2_hw,
        "ARIMA_R2": r2_arima
    }

    return model_scaled, model_pca, model_ridge, model_lasso, model_hw, model_arima, metrics