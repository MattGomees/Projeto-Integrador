import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import warnings

warnings.simplefilter('ignore', UserWarning)

# Ordem oficial das features para garantir consistência matricial entre Treino e Teste
FEATURES_ORDER = ['time-5', 'time-4', 'time-3', 'time-2', 'time-1']

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
    Aplica StandardScaler. 
    NOTA RIGOROSA: O fit (cálculo de média/desvio) é feito APENAS no conjunto de treino.
    O conjunto de teste é apenas transformado usando os parâmetros aprendidos no treino.
    Isso evita Data Leakage (vazamento de dados futuros).
    """
    # --- ALTERAÇÃO DE SEGURANÇA: Ordem Explícita ---
    # Em vez de drop('time'), selecionamos as colunas na ordem correta.
    # Isso evita erros se o CSV vier com colunas trocadas.
    X = df[FEATURES_ORDER]
    y = df['time']
    
    # Split sem embaralhar (Série Temporal)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False 
    )

    # Ajusta scaler APENAS no treino
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    
    # Apenas transforma o teste (usando média/desvio do treino)
    X_test_scaled = scaler.transform(X_test_df)

    # Prepara dados full para treino final (Ajusta scaler no dataset todo para produção)
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X)

    return (
        (X_train_scaled, X_test_scaled, scaler),
        (y_train, y_test),
        (X_full_scaled, y),
        scaler_full 
    )

def get_expected_performance_sklearn(X_full_raw, y):
    """
    Calcula performance esperada com validação cruzada rigorosa (TimeSeriesSplit).
    O Scaler é ajustado DENTRO de cada fold para garantir independência estatística.
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
        X_test_fold_scaled = scaler_fold.transform(X_test_fold) # Apenas transform
        # ----------------------------------------------

        for model_name, model in models.items():
            from sklearn.base import clone
            clf = clone(model)
            
            clf.fit(X_train_fold_scaled, y_train_fold)
            try:
                y_pred = clf.predict(X_test_fold_scaled)
                
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

def preprocess_unseen_data(X_unseen, scaler):
    # IMPORTANTE: Nunca fazemos fit aqui. Apenas transform.
    X_unseen_scaled = scaler.transform(X_unseen)
    return X_unseen_scaled

def train_models(X_train_scaled, X_test_scaled, y_train, y_test):
    # Treina apenas os modelos lineares solicitados e compatíveis com a estrutura de input
    model_linear = train_sklearn_model(X_train_scaled, y_train, model_type='linear')
    model_ridge = train_sklearn_model(X_train_scaled, y_train, model_type='ridge')
    model_lasso = train_sklearn_model(X_train_scaled, y_train, model_type='lasso')
    model_elasticnet = train_sklearn_model(X_train_scaled, y_train, model_type='elasticnet')
    
    metrics = {
        "1_Linear_Std": {"R2": r2_score(y_test, model_linear.predict(X_test_scaled)), "RMSE": np.sqrt(mean_squared_error(y_test, model_linear.predict(X_test_scaled))), "MAE": mean_absolute_error(y_test, model_linear.predict(X_test_scaled))},
        "3_Ridge_L2": {"R2": r2_score(y_test, model_ridge.predict(X_test_scaled)), "RMSE": np.sqrt(mean_squared_error(y_test, model_ridge.predict(X_test_scaled))), "MAE": mean_absolute_error(y_test, model_ridge.predict(X_test_scaled))},
        "4_Lasso_L1": {"R2": r2_score(y_test, model_lasso.predict(X_test_scaled)), "RMSE": np.sqrt(mean_squared_error(y_test, model_lasso.predict(X_test_scaled))), "MAE": mean_absolute_error(y_test, model_lasso.predict(X_test_scaled))},
        "5_ElasticNet": {"R2": r2_score(y_test, model_elasticnet.predict(X_test_scaled)), "RMSE": np.sqrt(mean_squared_error(y_test, model_elasticnet.predict(X_test_scaled))), "MAE": mean_absolute_error(y_test, model_elasticnet.predict(X_test_scaled))},
    }

    return model_linear, model_ridge, model_lasso, model_elasticnet, metrics