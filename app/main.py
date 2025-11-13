# app/main.py

import traceback
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.responses import JSONResponse, StreamingResponse

# Importa as lógicas dos nossos outros módulos
# (Os imports relativos ESTÃO CORRETOS, pois temos o __init__.py)
from . import model_utils
from . import security_utils

# --- Constantes do Modelo ---
# Precisamos definir o alvo e as features aqui
# para que o main possa passá-los para as funções do model_utils.
TARGET_COL = "time"
# Deixe como None para que a função _select_features use todas as outras colunas
FEATURE_COLS = None 

# Cria a aplicação FastAPI
app = FastAPI(
    title="PI - API de Regressão de Séries Temporais",
    description="API para treinar, testar e avaliar um modelo de Regressão Linear.",
    version="1.0.0"
)

# --- Endpoint de Boas-vindas (Raiz) ---
@app.get("/", tags=["Root"])
async def read_root():
    """Endpoint raiz para verificar se a API está online."""
    return {"message": "Bem-vindo à API do Projeto Integrador. Visite /docs para detalhes."}

# --- Endpoint de Treinamento ---
@app.post("/train", tags=["Modelo"])
async def train_model(file: UploadFile = File(...)):
    """
    Endpoint para treinar o modelo.
    Recebe um arquivo .zip contendo o .csv de treino.
    Retorna o JSON com a expectativa de desempenho.
    """
    try:
        # 1. Descompacta o CSV do ZIP em memória
        df_train = security_utils.unzip_csv_from_upload(file)

        # 2. Chama a lógica de cálculo da expectativa (com o nome NOVO)
        expected_metrics = model_utils.timeseries_expected_performance(
            df_train=df_train,
            target_col=TARGET_COL,
            feature_cols=FEATURE_COLS
        )

        # 3. Chama a lógica de treino final (com o nome NOVO)
        model_utils.train_full_model(
            df_train=df_train,
            target_col=TARGET_COL,
            feature_cols=FEATURE_COLS
        )
        
        # 4. Retorna o resumo da expectativa
        return JSONResponse(
            status_code=200,
            content={
                "message": "Modelo treinado com sucesso.",
                "expected_performance": expected_metrics 
            }
        )
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro interno no servidor: {e}")

# --- Endpoint de Teste/Predição ---
@app.post("/test", tags=["Modelo"])
async def test_model(file: UploadFile = File(...)):
    """
    Endpoint para testar o modelo ou fazer predições.
    Recebe um .zip contendo o .csv de teste (com ou sem rótulos).
    """
    try:
        # 1. Descompacta o CSV do ZIP
        df_test = security_utils.unzip_csv_from_upload(file)

        # 2. Carrega os artefatos do modelo (pipeline, features, etc.)
        artifacts = model_utils.load_model_artifacts()

        # 3. Chama a lógica de predição (com o nome NOVO)
        y_pred_series = model_utils.predict_dataframe(df_test, artifacts)

        # 4. Constrói o DataFrame de resposta
        predictions_df = df_test.copy()
        predictions_df["y_pred"] = y_pred_series

        # 5. LÓGICA DE COM/SEM RÓTULOS (Agora vive no main.py)
        metrics_real = None
        if TARGET_COL in predictions_df.columns:
            print("Coluna alvo encontrada. Calculando métricas reais...")
            y_true = predictions_df[TARGET_COL]
            # Chama a função de avaliação
            metrics_real = model_utils.evaluate_predictions(y_true, y_pred_series)
        
        # 6. Compacta o DataFrame de predições em um .zip
        zip_response = security_utils.zip_csv_to_response(predictions_df)

        # 7. Adiciona as métricas (se existirem) no cabeçalho
        if metrics_real:
            metrics_json = json.dumps(metrics_real)
            zip_response.headers["X-Metrics-Real"] = metrics_json

        return zip_response

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro interno no servidor: {e}")

# --- Endpoint de Reset ---
@app.post("/reset", tags=["Modelo"])
async def reset_model():
    """
    Endpoint para resetar o modelo.
    (Esta função tinha o nome correto!)
    """
    try:
        model_utils.reset_artifacts()
        return JSONResponse(
            status_code=200,
            content={"message": "Modelo e artefatos resetados com sucesso."}
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro interno no servidor: {e}")