import os
import uvicorn
import io
import json
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from starlette.responses import JSONResponse, FileResponse, HTMLResponse
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from statsmodels.tsa.holtwinters.results import HoltWintersResults
from statsmodels.tsa.arima.model import ARIMAResults
from azure.storage.blob import BlobServiceClient 

from . import model_utils
from . import security_utils

app = FastAPI(
    title="PI - Laboratório de Modelos (Refinado)",
    description="API Otimizada: Treina 6 modelos, seleciona o melhor e aplica somente ele.",
    version="3.1.0", 
    docs_url="/dev-docs"
)

ENCRYPTION_KEY = "minha-chave-pi-123"

AZURE_STORAGE_CONNECTION_STRING = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
if not AZURE_STORAGE_CONNECTION_STRING:
    print("ALERTA: Usando fallback local (Azurite).")
    AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"

BLOB_CONTAINER_NAME = "modelos"

MODEL_LINEAR_PATH = "model_linear.pkl"
MODEL_RIDGE_PATH = "model_ridge.pkl"
MODEL_LASSO_PATH = "model_lasso.pkl"
MODEL_ELASTICNET_PATH = "model_elasticnet.pkl"
MODEL_HW_PATH = "model_hw.pkl"
MODEL_ARIMA_PATH = "model_arima.pkl"
SCALER_PATH = "scaler_std.pkl"
BEST_MODEL_INFO_PATH = "best_model_info.json"

MODEL_FILE_MAP = {
    "1_Linear_Std": MODEL_LINEAR_PATH,
    "3_Ridge_L2": MODEL_RIDGE_PATH,
    "4_Lasso_L1": MODEL_LASSO_PATH,
    "5_ElasticNet": MODEL_ELASTICNET_PATH,
    "6_HoltWinters": MODEL_HW_PATH,
    "7_ARIMA": MODEL_ARIMA_PATH
}

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def _get_blob_service():
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    try:
        container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
        container_client.get_container_properties() 
    except Exception:
        try:
            blob_service_client.create_container(BLOB_CONTAINER_NAME)
        except Exception as create_error:
            if 'ContainerAlreadyExists' not in str(create_error):
                raise
    return blob_service_client

def _save_artifact_to_blob(artifact_data, blob_name: str):
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
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob=blob_name)
    downloader = blob_client.download_blob()
    return downloader.readall()

@app.get("/", response_class=HTMLResponse, tags=["Interface"])
async def get_frontend_interface():
    return HTMLResponse(content="<h1>API PI Online</h1>", status_code=200)

@app.post("/train", tags=["Modelo"])
async def train(file: UploadFile = File(...)):
    train_file_path = os.path.join(UPLOAD_FOLDER, 'train.csv')
    try:
        with open(train_file_path, "wb") as buffer:
            buffer.write(await file.read())

        df = model_utils.load_data(train_file_path)

        # Recebe o scaler_full atualizado aqui
        (X_train_scaled, X_test_scaled, _), \
        (y_train, y_test), \
        (X_full_scaled, y_full), \
        scaler_full = model_utils.preprocess_data(df) # <--- UPDATED

        # 1. Treinamento e Validação
        (model_linear_val, model_ridge_val, model_lasso_val, model_elasticnet_val, model_hw_val, model_arima_val, metrics_val) = \
            model_utils.train_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # 2. Treinamento Final (Produção) nos Dados Completos 
        model_linear = model_utils.train_sklearn_model(X_full_scaled, y_full, model_type='linear')
        model_ridge = model_utils.train_sklearn_model(X_full_scaled, y_full, model_type='ridge')
        model_lasso = model_utils.train_sklearn_model(X_full_scaled, y_full, model_type='lasso')
        model_elasticnet = model_utils.train_sklearn_model(X_full_scaled, y_full, model_type='elasticnet') 
        
        model_hw = model_utils.train_holtwinters_model(y_full, y_full)[0] 
        model_arima = model_utils.train_arima_model(y_full, y_full)[0]

        # --- SALVANDO ARTEFATOS ---
        # Salva Scaler Full
        buf_s = io.BytesIO()
        joblib.dump(scaler_full, buf_s) # <--- USA O SCALER FULL
        buf_s.seek(0)
        _save_artifact_to_blob(buf_s.read(), SCALER_PATH)

        artifacts_sklearn = [
            (model_linear, MODEL_LINEAR_PATH),
            (model_ridge, MODEL_RIDGE_PATH),
            (model_lasso, MODEL_LASSO_PATH),
            (model_elasticnet, MODEL_ELASTICNET_PATH)
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

        # --- SELEÇÃO DO MELHOR MODELO (MCDA) ---
        # Passa X raw para a funcao agora, pois ela trata o scaler internamente
        X_raw = df.drop('time', axis=1)
        expected_performance_sklearn = model_utils.get_expected_performance_sklearn(X_raw, y_full)
        
        final_metrics = {}
        final_metrics["1_Linear_Std"] = expected_performance_sklearn["linear"]
        final_metrics["3_Ridge_L2"] = expected_performance_sklearn["ridge"]
        final_metrics["4_Lasso_L1"] = expected_performance_sklearn["lasso"]
        final_metrics["5_ElasticNet"] = expected_performance_sklearn["elasticnet"]
        final_metrics["6_HoltWinters"] = metrics_val["6_HoltWinters"]
        final_metrics["7_ARIMA"] = metrics_val["7_ARIMA"]

        df_rank = pd.DataFrame.from_dict(final_metrics, orient='index')
        df_rank['Rank_R2'] = df_rank['R2'].rank(ascending=False)
        df_rank['Rank_RMSE'] = df_rank['RMSE'].rank(ascending=True)
        df_rank['Rank_MAE'] = df_rank['MAE'].rank(ascending=True)
        df_rank['Total_Rank'] = df_rank[['Rank_R2', 'Rank_RMSE', 'Rank_MAE']].sum(axis=1)

        best_model_name = df_rank['Total_Rank'].idxmin()
        final_metrics["best_model_tech_name"] = best_model_name

        best_info = {"name": best_model_name}
        _save_artifact_to_blob(json.dumps(best_info).encode('utf-8'), BEST_MODEL_INFO_PATH)

        return {
            "message": f"Treino concluído. Melhor modelo selecionado: {best_model_name}",
            "full_expected_performance_metrics": final_metrics
        }
    except Exception as e:
        import traceback
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=f"Erro no treinamento: {str(e)}")
    finally:
        if os.path.exists(train_file_path): os.remove(train_file_path)

@app.post("/predict", tags=["Modelo"])
async def predict(file: UploadFile = File(...)):
    test_file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    results_csv_path = os.path.join(UPLOAD_FOLDER, 'predictions_best.csv')

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
                raise HTTPException(status_code=400, detail=f"CSV deve conter: {cols_needed}")
            X_unseen = df_raw[cols_needed]
            y_true = None

        try:
            info_bytes = _load_artifact_from_blob(BEST_MODEL_INFO_PATH)
            best_info = json.loads(info_bytes.decode('utf-8'))
            best_model_name = best_info["name"]
        except:
            raise HTTPException(status_code=400, detail="Modelo não treinado. Execute /train primeiro.")

        scaler = joblib.load(io.BytesIO(_load_artifact_from_blob(SCALER_PATH)))
        
        model_file_name = MODEL_FILE_MAP.get(best_model_name)
        if not model_file_name:
            raise HTTPException(status_code=500, detail=f"Arquivo para modelo {best_model_name} não mapeado.")

        artifact_bytes = _load_artifact_from_blob(model_file_name)
        
        predictions = []
        
        if best_model_name in ["1_Linear_Std", "3_Ridge_L2", "4_Lasso_L1", "5_ElasticNet"]:
            model = joblib.load(io.BytesIO(artifact_bytes))
            X_unseen_scaled = model_utils.preprocess_unseen_data(X_unseen, scaler)
            predictions = model.predict(X_unseen_scaled)
            
        elif best_model_name == "6_HoltWinters":
            try:
                model = HoltWintersResults.load(io.BytesIO(artifact_bytes))
            except:
                model = joblib.load(io.BytesIO(artifact_bytes))
            pred_raw = model.forecast(steps=len(X_unseen))
            predictions = pred_raw.values if hasattr(pred_raw, 'values') else pred_raw

        elif best_model_name == "7_ARIMA":
            model = ARIMAResults.load(io.BytesIO(artifact_bytes))
            pred_raw = model.forecast(steps=len(X_unseen))
            predictions = pred_raw.values if hasattr(pred_raw, 'values') else pred_raw

        results_df = pd.DataFrame({
            f'predicted_{best_model_name}': predictions
        })

        if has_labels and y_true is not None:
            results_df['true_time'] = y_true

        results_df.to_csv(results_csv_path, index=False)

        secured_data_path, secured_freq_path = security_utils.secure_file(
            results_csv_path, os.path.join(UPLOAD_FOLDER, 'predictions_secured'), ENCRYPTION_KEY
        )

        performance_report = {}
        if has_labels:
            performance_report[best_model_name] = {
                "R2_Score": r2_score(y_true, predictions),
                "RMSE": np.sqrt(mean_squared_error(y_true, predictions)),
                "MAE": mean_absolute_error(y_true, predictions)
            }
        else:
            performance_report = "Sem rótulos, avaliação não realizada."

        return {
            "status": "success",
            "best_model_used": best_model_name,
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
        print(f"ERRO: {e}")
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")
    finally:
        # FIX CRÍTICO DO 404:
        # Só deletamos o arquivo de input (upload), NÃO deletamos o output que o usuário vai baixar.
        if os.path.exists(test_file_path): os.remove(test_file_path)
        # REMOVIDO: if os.path.exists(results_csv_path): os.remove(results_csv_path)

@app.get("/download", tags=["Modelo"])
async def download(file: str = Query(..., description="Nome do arquivo")):
    file_path = os.path.join(UPLOAD_FOLDER, file)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='application/octet-stream', filename=file)
    else:
        raise HTTPException(status_code=404, detail="Arquivo não encontrado.")

@app.post("/reset", tags=["Modelo"])
async def reset_model():
    try:
        blob_service_client = _get_blob_service()
        container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

        artifacts = [
            MODEL_LINEAR_PATH, MODEL_RIDGE_PATH, MODEL_LASSO_PATH,
            MODEL_ELASTICNET_PATH, MODEL_HW_PATH, MODEL_ARIMA_PATH,
            SCALER_PATH, BEST_MODEL_INFO_PATH
        ]
        count = 0
        for blob_name in artifacts:
            try:
                container_client.delete_blob(blob_name)
                count += 1
            except: pass
        return {"message": f"Reset concluído. {count} artefatos removidos."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao resetar: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)