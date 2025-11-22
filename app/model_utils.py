import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)

# REMOVIDO: _winsorize_features (Causava distorção em séries temporais)

def load_data(file_path):
    df = pd.read_csv(file_path)
    if 'time' not in df.columns:
        return df
    # Garante que time é numérico e remove NaNs gerados por conversão
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df.dropna()
    return df

def preprocess_data(df, test_size=0.2):
    """
    Aplica StandardScaler. Winsorization removida para preservar tendência.
    """
    X = df.drop('time', axis=1)
    y = df['time']
    
    # Split sem embaralhar (Série Temporal)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False 
    )

    # Ajusta scaler APENAS no treino
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)

    # Prepara dados full para treino final (Ajusta scaler no dataset todo)
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X)

    # Retorna scaler_full para ser salvo e usado na produção
    return (
        (X_train_scaled, X_test_scaled, scaler), # Scaler do split (não usado para salvar, só para validar se quisesse)
        (y_train, y_test),
        (X_full_scaled, y),
        scaler_full # <--- IMPORTANTE: Retornamos o scaler treinado no FULL
    )

def get_expected_performance_sklearn(X_full_raw, y):
    """
    Calcula performance esperada com validação cruzada rigorosa.
    O Scaler é ajustado DENTRO de cada fold para evitar Data Leakage.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    
    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.1),
        "elasticnet": ElasticNet(alpha=0.1, l1_ratio=0.5)
    }
    
    # Converte para numpy para indexação fácil
    X_arr = X_full_raw.values if hasattr(X_full_raw, 'values') else X_full_raw
    y_arr = y.values if hasattr(y, 'values') else y
    
    results = { name: {'R2': [], 'RMSE': [], 'MAE': []} for name in models }

    for train_index, test_index in tscv.split(X_arr):
        X_train_fold, X_test_fold = X_arr[train_index], X_arr[test_index]
        y_train_fold, y_test_fold = y_arr[train_index], y_arr[test_index]
        
        # --- RIGOROSO: Fit do Scaler dentro do Fold ---
        scaler_fold = StandardScaler()
        X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold)
        X_test_fold_scaled = scaler_fold.transform(X_test_fold)
        # ----------------------------------------------

        for model_name, model in models.items():
            # Clona o modelo para garantir que está limpo
            from sklearn.base import clone
            clf = clone(model)
            
            clf.fit(X_train_fold_scaled, y_train_fold)
            try:
                y_pred = clf.predict(X_test_fold_scaled)
                
                # Cálculo seguro das métricas
                r2 = r2_score(y_test_fold, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred))
                mae = mean_absolute_error(y_test_fold, y_pred)
                
                results[model_name]['R2'].append(r2)
                results[model_name]['RMSE'].append(rmse)
                results[model_name]['MAE'].append(mae)
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

def train_sklearn_model(X_train_scaled, y_train, model_type='linear'):
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0) 
    elif model_type == 'lasso':
        model = Lasso(alpha=0.1)
    elif model_type == 'elasticnet': 
        model = ElasticNet(alpha=0.1, l1_ratio=0.5) 
    
    model.fit(X_train_scaled, y_train)
    return model

def train_holtwinters_model(y_train, y_test):
    best_aic = float('inf')
    best_model = None
    configs = [
        {'trend': 'add', 'seasonal': None},
        {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 5},
        {'trend': None, 'seasonal': 'add', 'seasonal_periods': 5}
    ]
    
    # Garante que índice é numérico/range para evitar erros do statsmodels
    y_train_reset = y_train.reset_index(drop=True)
    
    for config in configs:
        try:
            model = ExponentialSmoothing(
                y_train_reset, 
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
        best_model = ExponentialSmoothing(y_train_reset).fit()
    
    steps = len(y_test) if len(y_test) > 0 else 1
    y_pred_hw = best_model.forecast(steps=steps)
    
    if len(y_test) > 0:
        expected_r2 = r2_score(y_test, y_pred_hw)
        expected_rmse = np.sqrt(mean_squared_error(y_test, y_pred_hw))
        expected_mae = mean_absolute_error(y_test, y_pred_hw)
    else:
        expected_r2, expected_rmse, expected_mae = 0, 0, 0

    return best_model, expected_r2, expected_rmse, expected_mae

def train_arima_model(y_train, y_test):
    best_aic = float('inf')
    best_model_fit = None
    # Grid reduzido para performance e robustez
    p_values = [1, 2]
    d_values = [0, 1]
    q_values = [0, 1]
    
    y_train_reset = y_train.reset_index(drop=True)

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(y_train_reset, order=(p, d, q))
                    model_fit = model.fit()
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_model_fit = model_fit
                except:
                    continue
    if best_model_fit is None:
        best_model_fit = ARIMA(y_train_reset, order=(1, 1, 1)).fit()

    steps = len(y_test) if len(y_test) > 0 else 1
    y_pred_arima = best_model_fit.forecast(steps=steps)
    
    if len(y_test) > 0:
        expected_r2 = r2_score(y_test, y_pred_arima)
        expected_rmse = np.sqrt(mean_squared_error(y_test, y_pred_arima))
        expected_mae = mean_absolute_error(y_test, y_pred_arima)
    else:
        expected_r2, expected_rmse, expected_mae = 0, 0, 0
    
    return best_model_fit, expected_r2, expected_rmse, expected_mae

def preprocess_unseen_data(X_unseen, scaler):
    X_unseen_scaled = scaler.transform(X_unseen)
    return X_unseen_scaled

def train_models(X_train_scaled, X_test_scaled, y_train, y_test):
    model_linear = train_sklearn_model(X_train_scaled, y_train, model_type='linear')
    model_ridge = train_sklearn_model(X_train_scaled, y_train, model_type='ridge')
    model_lasso = train_sklearn_model(X_train_scaled, y_train, model_type='lasso')
    model_elasticnet = train_sklearn_model(X_train_scaled, y_train, model_type='elasticnet')
    
    model_hw, r2_hw, rmse_hw, mae_hw = train_holtwinters_model(y_train, y_test)
    model_arima, r2_arima, rmse_arima, mae_arima = train_arima_model(y_train, y_test)
    
    metrics = {
        "1_Linear_Std": {"R2": r2_score(y_test, model_linear.predict(X_test_scaled)), "RMSE": np.sqrt(mean_squared_error(y_test, model_linear.predict(X_test_scaled))), "MAE": mean_absolute_error(y_test, model_linear.predict(X_test_scaled))},
        "3_Ridge_L2": {"R2": r2_score(y_test, model_ridge.predict(X_test_scaled)), "RMSE": np.sqrt(mean_squared_error(y_test, model_ridge.predict(X_test_scaled))), "MAE": mean_absolute_error(y_test, model_ridge.predict(X_test_scaled))},
        "4_Lasso_L1": {"R2": r2_score(y_test, model_lasso.predict(X_test_scaled)), "RMSE": np.sqrt(mean_squared_error(y_test, model_lasso.predict(X_test_scaled))), "MAE": mean_absolute_error(y_test, model_lasso.predict(X_test_scaled))},
        "5_ElasticNet": {"R2": r2_score(y_test, model_elasticnet.predict(X_test_scaled)), "RMSE": np.sqrt(mean_squared_error(y_test, model_elasticnet.predict(X_test_scaled))), "MAE": mean_absolute_error(y_test, model_elasticnet.predict(X_test_scaled))},
        "6_HoltWinters": {"R2": r2_hw, "RMSE": rmse_hw, "MAE": mae_hw},
        "7_ARIMA": {"R2": r2_arima, "RMSE": rmse_arima, "MAE": mae_arima}
    }

    return model_linear, model_ridge, model_lasso, model_elasticnet, model_hw, model_arima, metrics