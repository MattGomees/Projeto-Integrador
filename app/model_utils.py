import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.decomposition import PCA
import os
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA


def _winsorize_features(df, lower_quantile=0.01, upper_quantile=0.99):
    df_copy = df.copy()
    feature_cols = [col for col in df_copy.columns if col.startswith('time-')]
    
    for col in feature_cols:
        if df_copy[col].dtype in ['float64', 'int64']:
            lower_bound = df_copy[col].quantile(lower_quantile)
            upper_bound = df_copy[col].quantile(upper_quantile)
            df_copy[col] = np.clip(df_copy[col], lower_bound, upper_bound)
    return df_copy

def load_data(file_path):
    df = pd.read_csv(file_path)
    if 'time' not in df.columns:
        return df
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df.dropna()
    return df

def preprocess_data(df, test_size=0.2):
    df_clean = _winsorize_features(df)

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
        "lasso": Lasso(alpha=0.1),
	"elasticnet": ElasticNet(alpha=0.1, l1_ratio=0.5)
    }
    
    data_X = {
        "linear_minmax": X_full_scaled_minmax,
        "linear_pca": X_full_pca,
        "ridge": X_full_scaled_std,
        "lasso": X_full_scaled_std,
	"elasticnet": X_full_scaled_std
    }
    
    results = { name: {'R2': [], 'RMSE': [], 'MAE': []} for name in models }

    for model_name in models:
        for train_index, test_index in tscv.split(data_X[model_name]):
            X_train, X_test = data_X[model_name][train_index], data_X[model_name][test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model = models[model_name]
            model.fit(X_train, y_train)
            try:
                y_pred = model.predict(X_test)
                
                results[model_name]['R2'].append(r2_score(y_test, y_pred))
                results[model_name]['RMSE'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
                results[model_name]['MAE'].append(mean_absolute_error(y_test, y_pred))

            except:
                 results[model_name]['R2'].append(np.nan) 
                 results[model_name]['RMSE'].append(np.nan)
                 results[model_name]['MAE'].append(np.nan)

    return {
        name: {
            'R2': np.nanmean(results[name]['R2']),
            'RMSE': np.nanmean(results[name]['RMSE']),
            'MAE': np.nanmean(results[name]['MAE'])
        } 
        for name in models
    }

def train_sklearn_model(X_train, y_train, model_type='linear'):
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0) 
    elif model_type == 'lasso':
        model = Lasso(alpha=0.1)
    elif model_type == 'elasticnet': 
        model = ElasticNet(alpha=0.1, l1_ratio=0.5) 
    
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
    expected_rmse = np.sqrt(mean_squared_error(y_test, y_pred_hw))
    expected_mae = mean_absolute_error(y_test, y_pred_hw)

    return best_model, expected_r2, expected_rmse, expected_mae


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
    expected_rmse = np.sqrt(mean_squared_error(y_test, y_pred_arima))
    expected_mae = mean_absolute_error(y_test, y_pred_arima)
    
    return best_model_fit, expected_r2, expected_rmse, expected_mae

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
    model_pca = train_sklearn_model(X_train_pca, y_train, model_type='linear')
    model_ridge = train_sklearn_model(X_train_scaled_std, y_train, model_type='ridge')
    model_lasso = train_sklearn_model(X_train_scaled_std, y_train, model_type='lasso')
    model_elasticnet = train_sklearn_model(X_train_scaled_std, y_train, model_type='elasticnet')
    
    model_hw, r2_hw, rmse_hw, mae_hw = train_holtwinters_model(y_train, y_test)
    model_arima, r2_arima, rmse_arima, mae_arima = train_arima_model(y_train, y_test)
    
    metrics = {
        "1_Linear_MinMax": {"R2": r2_score(y_test, model_scaled.predict(X_test_scaled)), "RMSE": np.sqrt(mean_squared_error(y_test, model_scaled.predict(X_test_scaled))), "MAE": mean_absolute_error(y_test, model_scaled.predict(X_test_scaled))},
        "2_Linear_PCA": {"R2": r2_score(y_test, model_pca.predict(X_test_pca)), "RMSE": np.sqrt(mean_squared_error(y_test, model_pca.predict(X_test_pca))), "MAE": mean_absolute_error(y_test, model_pca.predict(X_test_pca))},
        "3_Ridge_L2": {"R2": r2_score(y_test, model_ridge.predict(X_test_scaled_std)), "RMSE": np.sqrt(mean_squared_error(y_test, model_ridge.predict(X_test_scaled_std))), "MAE": mean_absolute_error(y_test, model_ridge.predict(X_test_scaled_std))},
        "4_Lasso_L1": {"R2": r2_score(y_test, model_lasso.predict(X_test_scaled_std)), "RMSE": np.sqrt(mean_squared_error(y_test, model_lasso.predict(X_test_scaled_std))), "MAE": mean_absolute_error(y_test, model_lasso.predict(X_test_scaled_std))},
        "5_ElasticNet": {"R2": r2_score(y_test, model_elasticnet.predict(X_test_scaled_std)), "RMSE": np.sqrt(mean_squared_error(y_test, model_elasticnet.predict(X_test_scaled_std))), "MAE": mean_absolute_error(y_test, model_elasticnet.predict(X_test_scaled_std))}, # Adicionado
        "6_HoltWinters": {"R2": r2_hw, "RMSE": rmse_hw, "MAE": mae_hw},
        "7_ARIMA": {"R2": r2_arima, "RMSE": rmse_arima, "MAE": mae_arima}
    }

    return model_scaled, model_pca, model_ridge, model_lasso, model_elasticnet, model_hw, model_arima, metrics
