import os
import uvicorn
import io
import json
import pandas as pd
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from starlette.responses import JSONResponse, FileResponse, HTMLResponse
from sklearn.metrics import r2_score, mean_squared_error

# Imports específicos para carregar corretamente os modelos de Time Series
from statsmodels.tsa.holtwinters.results import HoltWintersResults
from statsmodels.tsa.arima.model import ARIMAResults

# Nossos módulos locais
import model_utils
import security_utils

from azure.storage.blob import BlobServiceClient

app = FastAPI(
    title="PI - Laboratório de Modelos de Regressão",
    description="API que treina 6 modelos de 4 matérias para prever séries temporais.",
    version="2.0.2", 
    docs_url="/dev-docs"
)

ENCRYPTION_KEY = "minha-chave-pi-123"

# --- Conexão com Azure (Prof. Denis) ---
AZURE_STORAGE_CONNECTION_STRING = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
if not AZURE_STORAGE_CONNECTION_STRING:
    print("ALERTA: AZURE_STORAGE_CONNECTION_STRING não definida. Usando fallback local (Azurite).")

    AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"

BLOB_CONTAINER_NAME = "modelos"

# Caminhos dos artefatos
MODEL_SCALED_PATH = "model_scaled.pkl"
MODEL_PCA_PATH = "model_pca.pkl"
MODEL_RIDGE_PATH = "model_ridge.pkl"
MODEL_LASSO_PATH = "model_lasso.pkl"
MODEL_HW_PATH = "model_hw.pkl"
MODEL_ARIMA_PATH = "model_arima.pkl"

SCALER_MINMAX_PATH = "scaler_minmax.pkl"
SCALER_STD_PATH = "scaler_std.pkl"
PCA_PATH = "pca.pkl"

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Funções Auxiliares de Nuvem ---
def _get_blob_service():
    """Retorna o cliente do serviço de Blob, criando o container se não existir."""
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    try:
        container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
        if not container_client.exists():
             blob_service_client.create_container(BLOB_CONTAINER_NAME)
    except Exception:
        
        try:
            blob_service_client.create_container(BLOB_CONTAINER_NAME)
        except:
            pass
    return blob_service_client

def _save_artifact_to_blob(artifact_data, blob_name: str):
    """Serializa um artefato e envia para o Blob."""
    blob_service_client = _get_blob_service()
    blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob=blob_name)

    
    if isinstance(artifact_data, bytes):
        blob_client.upload_blob(artifact_data, overwrite=True)
    else:
  
        buf = io.BytesIO()
        joblib.dump(artifact_data, buf)
        buf.seek(0)
        blob_client.upload_blob(buf.read(), overwrite=True)

def _load_artifact_from_blob(blob_name: str) -> bytes:
    """Baixa um artefato em bytes do Blob Storage."""
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob=blob_name)
    downloader = blob_client.download_blob()
    return downloader.readall()



# ROTA DE INTERFACE

@app.get("/", response_class=HTMLResponse, tags=["Interface"])
async def get_frontend_interface():
    html_file_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_file_path):
        with open(html_file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    else:
        return HTMLResponse(content="<h1>API do Projeto Integrador Online</h1><p>Use o Streamlit para interagir.</p>", status_code=200)



# ENDPOINT: TREINO

@app.post("/train", tags=["Modelo"])
async def train(file: UploadFile = File(...)):
    """
    Treina todos os 6 modelos com o arquivo .csv enviado e os salva na nuvem.
    """
    train_file_path = os.path.join(UPLOAD_FOLDER, 'train.csv')
    try:
        # Salva o arquivo temporariamente
        with open(train_file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Carrega e processa
        df = model_utils.load_data(train_file_path)

        (X_train_scaled, X_test_scaled, scaler_minmax), \
        (X_train_pca, X_test_pca, scaler_std, pca), \
        (X_train_scaled_std, X_test_scaled_std, _), \
        (y_train, y_test), \
        (X_full_scaled_minmax, X_full_pca, X_full_scaled_std, y_full) = model_utils.preprocess_data(df)

        # Treinamento
        model_scaled, model_pca, model_ridge, model_lasso, model_hw, model_arima, metrics = \
            model_utils.train_models(
                X_train_scaled, X_test_scaled,
                X_train_pca, X_test_pca,
                X_train_scaled_std, X_test_scaled_std,
                y_train, y_test
            )

        # --- SALVANDO ARTEFATOS ---
        artifacts_sklearn = [
            (model_scaled, MODEL_SCALED_PATH),
            (model_pca, MODEL_PCA_PATH),
            (model_ridge, MODEL_RIDGE_PATH),
            (model_lasso, MODEL_LASSO_PATH),
            (scaler_minmax, SCALER_MINMAX_PATH),
            (scaler_std, SCALER_STD_PATH),
            (pca, PCA_PATH)
        ]

        for artifact, name in artifacts_sklearn:
            buf = io.BytesIO()
            joblib.dump(artifact, buf)
            buf.seek(0)
            _save_artifact_to_blob(buf.read(), name)

        buf_hw = io.BytesIO()
        model_hw.save(buf_hw)
        buf_hw.seek(0)
        _save_artifact_to_blob(buf_hw.read(), MODEL_HW_PATH)

        buf_arima = io.BytesIO()
        model_arima.save(buf_arima)
        buf_arima.seek(0)
        _save_artifact_to_blob(buf_arima.read(), MODEL_ARIMA_PATH)

        
        expected_r2_scaled, expected_r2_pca, expected_r2_ridge, expected_r2_lasso = \
            model_utils.get_expected_performance_sklearn(X_full_scaled_minmax, X_full_pca, X_full_scaled_std, y_full)


        expected_r2_hw = metrics["HoltWinters_R2"]
        expected_r2_arima = metrics["ARIMA_R2"]

        return {
            "message": "Modelos treinados com sucesso e salvos no Azure Blob Storage.",
            "expected_performance_R2_comparison": {
                "1_Linear_MinMax": expected_r2_scaled,
                "2_Linear_PCA": expected_r2_pca,
                "3_Ridge_L2": expected_r2_ridge,
                "4_Lasso_L1": expected_r2_lasso,
                "5_HoltWinters": expected_r2_hw,
                "6_ARIMA": expected_r2_arima
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=f"Erro no treinamento: {str(e)}")
    finally:
        if os.path.exists(train_file_path):
            os.remove(train_file_path)


# ENDPOINT: PREDICT
@app.post("/predict", tags=["Modelo"])
async def predict(file: UploadFile = File(...)):
    test_file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    results_csv_path = os.path.join(UPLOAD_FOLDER, 'predictions_all_models.csv')

    try:
        with open(test_file_path, "wb") as buffer:
            buffer.write(await file.read())

        df_raw = pd.read_csv(test_file_path)
        has_labels = 'time' in df_raw.columns

        if has_labels:
            df = model_utils.load_data(test_file_path)
            X_unseen = df[['time-5', 'time-4', 'time-3', 'time-2', 'time-1']]
            y_true = df['time'].values
        else:
            cols_needed = ['time-5', 'time-4', 'time-3', 'time-2', 'time-1']
            if not set(cols_needed).issubset(df_raw.columns):
                raise HTTPException(
                    status_code=400,
                    detail=f"CSV deve conter as colunas: {cols_needed}"
                )
            X_unseen = df_raw[cols_needed]
            y_true = None

        # --- Carregando do Blob ---
        # Scikit-learn
        model_scaled = joblib.load(io.BytesIO(_load_artifact_from_blob(MODEL_SCALED_PATH)))
        model_pca = joblib.load(io.BytesIO(_load_artifact_from_blob(MODEL_PCA_PATH)))
        model_ridge = joblib.load(io.BytesIO(_load_artifact_from_blob(MODEL_RIDGE_PATH)))
        model_lasso = joblib.load(io.BytesIO(_load_artifact_from_blob(MODEL_LASSO_PATH)))
        scaler_minmax = joblib.load(io.BytesIO(_load_artifact_from_blob(SCALER_MINMAX_PATH)))
        scaler_std = joblib.load(io.BytesIO(_load_artifact_from_blob(SCALER_STD_PATH)))
        pca_obj = joblib.load(io.BytesIO(_load_artifact_from_blob(PCA_PATH)))

        hw_bytes = _load_artifact_from_blob(MODEL_HW_PATH)
        try:
            model_hw = HoltWintersResults.load(io.BytesIO(hw_bytes))
        except:
             model_hw = joblib.load(io.BytesIO(hw_bytes))

        arima_bytes = _load_artifact_from_blob(MODEL_ARIMA_PATH)
        model_arima = ARIMAResults.load(io.BytesIO(arima_bytes))

        # --- Processar ---
        X_unseen_scaled_minmax, X_unseen_pca, X_unseen_scaled_std = model_utils.preprocess_unseen_data(
            X_unseen, scaler_minmax, scaler_std, pca_obj
        )

        # --- Predições ---
        predictions_scaled = model_scaled.predict(X_unseen_scaled_minmax)
        predictions_pca = model_pca.predict(X_unseen_pca)
        predictions_ridge = model_ridge.predict(X_unseen_scaled_std)
        predictions_lasso = model_lasso.predict(X_unseen_scaled_std)

        steps_to_forecast = len(X_unseen)
        
        # HoltWinters e ARIMA retornam Series ou Arrays dependendo da versão
        pred_hw_raw = model_hw.forecast(steps=steps_to_forecast)
        predictions_hw = pred_hw_raw.values if hasattr(pred_hw_raw, 'values') else pred_hw_raw
        
        pred_arima_raw = model_arima.forecast(steps=steps_to_forecast)
        predictions_arima = pred_arima_raw.values if hasattr(pred_arima_raw, 'values') else pred_arima_raw

        results_df = pd.DataFrame({
            'predicted_Linear_Scaled': predictions_scaled,
            'predicted_Linear_PCA': predictions_pca,
            'predicted_Ridge_L2': predictions_ridge,
            'predicted_Lasso_L1': predictions_lasso,
            'predicted_HoltWinters': predictions_hw,
            'predicted_ARIMA': predictions_arima
        })

        if has_labels and y_true is not None:
            results_df['true_time'] = y_true

        results_df.to_csv(results_csv_path, index=False)

        # Segurança
        secured_data_path, secured_freq_path = security_utils.secure_file(
            results_csv_path, os.path.join(UPLOAD_FOLDER, 'predictions_all_secured'), ENCRYPTION_KEY
        )

        # Avaliação
        if has_labels:
            performance_report = {
                "1_Linear_MinMax": {"R2_Score": r2_score(y_true, predictions_scaled)},
                "2_Linear_PCA": {"R2_Score": r2_score(y_true, predictions_pca)},
                "3.Ridge_L2": {"R2_Score": r2_score(y_true, predictions_ridge)},
                "4.Lasso_L1": {"R2_Score": r2_score(y_true, predictions_lasso)},
                "5_HoltWinters": {"R2_Score": r2_score(y_true, predictions_hw)},
                "6_ARIMA": {"R2_Score": r2_score(y_true, predictions_arima)}
            }
        else:
            performance_report = "Sem rótulos, avaliação não realizada."

        return {
            "status": "success",
            "resultado_compactado_path": secured_data_path,
            "tabela_de_frequencias_path": secured_freq_path,
            "performance": performance_report,
            "download_links": {
                 "csv": f"/download?file={os.path.basename(results_csv_path)}",
                 "huff": f"/download?file={os.path.basename(secured_data_path)}",
                 "freq": f"/download?file={os.path.basename(secured_freq_path)}"
            }
        }
    except Exception as e:
        print(f"ERRO NO PREDICT: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")
    finally:
        if os.path.exists(test_file_path):
            os.remove(test_file_path)


# ENDPOINT: DOWNLOAD

@app.get("/download", tags=["Modelo"])
async def download(file: str = Query(..., description="Nome do arquivo")):
    file_path = os.path.join(UPLOAD_FOLDER, file)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='application/octet-stream', filename=file)
    else:
        raise HTTPException(status_code=404, detail="Arquivo não encontrado.")


# ENDPOINT: RESET
@app.post("/reset", tags=["Modelo"])
async def reset_model():
    try:
        blob_service_client = _get_blob_service()
        container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

        artifacts = [
            MODEL_SCALED_PATH, MODEL_PCA_PATH, MODEL_RIDGE_PATH, MODEL_LASSO_PATH,
            MODEL_HW_PATH, MODEL_ARIMA_PATH,
            SCALER_MINMAX_PATH, SCALER_STD_PATH, PCA_PATH
        ]

        count = 0
        for blob_name in artifacts:
            try:
                container_client.delete_blob(blob_name)
                count += 1
            except Exception:
                pass

        return {"message": f"Reset concluído. {count} artefatos removidos."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao resetar: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)