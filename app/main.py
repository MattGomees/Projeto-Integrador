import os
import uvicorn 
import io      
import json
import pandas as pd
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from starlette.responses import JSONResponse, FileResponse, HTMLResponse
from sklearn.metrics import r2_score, mean_squared_error

# Nossos módulos locais
import model_utils
import security_utils

# Importações para o Azure Blob Storage (Prof. Denis)
from azure.storage.blob import BlobServiceClient

# Importações dos carregadores de modelo
from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tsa.holtwinters import HoltWintersResults
# --- REMOVIDO: Importação do load_model do Keras ---
# from tensorflow.keras.models import load_model

# --- Configuração Inicial ---
app = FastAPI(
    title="PI - Laboratório de Modelos de Regressão",
    description="API que treina 6 modelos de 4 matérias para prever séries temporais.",
    version="2.0.0",
    docs_url="/dev-docs" 
)

ENCRYPTION_KEY = "minha-chave-pi-123"

# --- Conexão com Azure (Prof. Denis) ---
AZURE_STORAGE_CONNECTION_STRING = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
if not AZURE_STORAGE_CONNECTION_STRING:
    print("ALERTA: AZURE_STORAGE_CONNECTION_STRING não definida. Usando fallback local (Azurite).")
    AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"

BLOB_CONTAINER_NAME = "modelos" 

# --- Nomes dos Artefatos ---
MODEL_SCALED_PATH = 'model_scaled.pkl' 
MODEL_PCA_PATH = 'model_pca.pkl'       
MODEL_RIDGE_PATH = 'model_ridge.pkl'   
MODEL_LASSO_PATH = 'model_lasso.pkl'   
MODEL_HW_PATH = 'model_holtwinters.pkl' 
MODEL_ARIMA_PATH = 'model_arima.pkl'    
# --- REMOVIDO: MODEL_LSTM_PATH ---
SCALER_MINMAX_PATH = 'scaler_minmax.pkl'
SCALER_STD_PATH = 'scaler_std.pkl'
PCA_PATH = 'pca.pkl'

# --- REMOVIDO: LSTM da lista de artefatos ---
ALL_ARTIFACT_PATHS = [
    MODEL_SCALED_PATH, MODEL_PCA_PATH, MODEL_RIDGE_PATH, MODEL_LASSO_PATH,
    MODEL_HW_PATH, MODEL_ARIMA_PATH,
    SCALER_MINMAX_PATH, SCALER_STD_PATH, PCA_PATH
]

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Funções Auxiliares de Nuvem ---
def _save_artifact_to_blob(artifact_data, blob_name: str):
    """Serializa um artefato (modelo/scaler) em memória e o envia para o Blob."""
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    try:
        blob_service_client.create_container(BLOB_CONTAINER_NAME)
    except Exception:
        pass 

    blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob=blob_name)
    
    if isinstance(artifact_data, bytes):
        blob_client.upload_blob(artifact_data, overwrite=True)
    else:
        artifact_data.seek(0)
        blob_client.upload_blob(artifact_data.read(), overwrite=True)

def _load_artifact_from_blob(blob_name: str) -> bytes:
    """Baixa os bytes de um artefato do Blob."""
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob=blob_name)
    
    if not blob_client.exists():
        raise FileNotFoundError(f"Artefato '{blob_name}' não encontrado no Blob Storage. Faça o /train primeiro.")
        
    downloader = blob_client.download_blob()
    return downloader.readall()

# --- Endpoints da API ---
@app.get("/", response_class=HTMLResponse, tags=["Interface"])
async def get_frontend_interface():
    """
    Serve a interface gráfica amigável (index.html).
    """
    html_file_path = os.path.join(os.path.dirname(__file__), 'index.html')
    if os.path.exists(html_file_path):
        return FileResponse(html_file_path)
    else:
        raise HTTPException(status_code=404, detail="Interface (index.html) não encontrada.")

@app.post("/train", tags=["Modelo"])
async def train(file: UploadFile = File(...)):
    """
    Treina todos os 6 modelos com o arquivo .csv enviado e os salva na nuvem.
    """
    train_file_path = os.path.join(UPLOAD_FOLDER, 'train.csv')
    try:
        with open(train_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        df = model_utils.load_data(train_file_path)
        
        (X_train_scaled, X_test_scaled, scaler_minmax), \
        (X_train_pca, X_test_pca, scaler_std, pca), \
        (X_train_scaled_std, X_test_scaled_std, _), \
        (y_train, y_test), \
        (X_full_scaled_minmax, X_full_pca, X_full_scaled_std, y_full) = model_utils.preprocess_data(df)

        # --- Treinamento ---
        model_scaled = model_utils.train_sklearn_model(X_train_scaled, y_train, model_type='linear')
        model_pca = model_utils.train_sklearn_model(X_train_pca, y_train, model_type='linear')
        model_ridge = model_utils.train_sklearn_model(X_train_scaled_std, y_train, model_type='ridge')
        model_lasso = model_utils.train_sklearn_model(X_train_scaled_std, y_train, model_type='lasso')
        model_hw, expected_r2_hw = model_utils.train_holtwinters_model(y_train, y_test)
        model_arima, expected_r2_arima = model_utils.train_arima_model(y_train, y_test)
        
        # --- REMOVIDO: Treino do LSTM ---
        
        # --- Persistência na Nuvem (PATCH 1 Corrigido) ---
        buf = io.BytesIO()
        joblib.dump(model_scaled, buf)
        _save_artifact_to_blob(buf, MODEL_SCALED_PATH)

        buf = io.BytesIO()
        joblib.dump(model_pca, buf)
        _save_artifact_to_blob(buf, MODEL_PCA_PATH)

        buf = io.BytesIO()
        joblib.dump(model_ridge, buf)
        _save_artifact_to_blob(buf, MODEL_RIDGE_PATH)

        buf = io.BytesIO()
        joblib.dump(model_lasso, buf)
        _save_artifact_to_blob(buf, MODEL_LASSO_PATH)

        buf = io.BytesIO()
        joblib.dump(scaler_minmax, buf)
        _save_artifact_to_blob(buf, SCALER_MINMAX_PATH)

        buf = io.BytesIO()
        joblib.dump(scaler_std, buf)
        _save_artifact_to_blob(buf, SCALER_STD_PATH)

        buf = io.BytesIO()
        joblib.dump(pca, buf)
        _save_artifact_to_blob(buf, PCA_PATH)
            
        with io.BytesIO() as buffer:
            model_hw.save(buffer)
            _save_artifact_to_blob(buffer, MODEL_HW_PATH)
            
        with io.BytesIO() as buffer:
            model_arima.save(buffer)
            _save_artifact_to_blob(buffer, MODEL_ARIMA_PATH)
            
        # --- REMOVIDO: Bloco de salvar o LSTM ---

        # --- Expectativa de Desempenho (Prof. Miro & Fernando) ---
        expected_r2_scaled, expected_r2_pca, expected_r2_ridge, expected_r2_lasso = \
            model_utils.get_expected_performance_sklearn(X_full_scaled_minmax, X_full_pca, X_full_scaled_std, y_full)

        return {
            "message": "Modelos treinados com sucesso e salvos no Azure Blob Storage.",
            "expected_performance_R2_comparison (Validação)": {
                "1_Linear_MinMax (PI_Base)": expected_r2_scaled,
                "2_Linear_PCA (Thais)": expected_r2_pca,
                "3_Ridge_L2 (Fernando)": expected_r2_ridge,
                "4.Lasso_L1 (Fernando)": expected_r2_lasso,
                "5_HoltWinters (Miro)": expected_r2_hw,
                "6_ARIMA (Miro)": expected_r2_arima
                # --- REMOVIDO: "7_LSTM (Miro)" ---
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no treinamento: {str(e)}")
    finally:
        if os.path.exists(train_file_path):
            os.remove(train_file_path)

@app.post("/predict", tags=["Modelo"])
async def predict(file: UploadFile = File(...)):
    """
    Recebe um .csv, executa a predição com os 6 modelos e retorna os
    resultados compactados (Huffman) e criptografados (XOR).
    """
    test_file_path = os.path.join(UPLOAD_FOLDER, file.filename) 
    results_csv_path = os.path.join(UPLOAD_FOLDER, 'predictions_all_models.csv') 
    
    try:
        with open(test_file_path, "wb") as buffer:
            buffer.write(await file.read())

        # --- Lógica (PATCH 2) para CSV com ou sem rótulo ---
        df_raw = pd.read_csv(test_file_path)
        has_labels = 'time' in df_raw.columns

        if has_labels:
            df_unseen = model_utils.load_data(test_file_path)
            y_true = df_unseen['time'] 
        else:
            df_unseen = df_raw.copy()
        
        X_unseen = df_unseen.drop('time', axis=1, errors='ignore')
        # --- FIM DO PATCH 2 ---

        # --- Carregar os 9 artefatos do Azure Blob Storage ---
        model_scaled = joblib.load(io.BytesIO(_load_artifact_from_blob(MODEL_SCALED_PATH)))
        model_pca = joblib.load(io.BytesIO(_load_artifact_from_blob(MODEL_PCA_PATH)))
        model_ridge = joblib.load(io.BytesIO(_load_artifact_from_blob(MODEL_RIDGE_PATH)))
        model_lasso = joblib.load(io.BytesIO(_load_artifact_from_blob(MODEL_LASSO_PATH)))
        model_hw = HoltWintersResults.load(io.BytesIO(_load_artifact_from_blob(MODEL_HW_PATH)))
        model_arima = ARIMAResults.load(io.BytesIO(_load_artifact_from_blob(MODEL_ARIMA_PATH)))
        
        # --- REMOVIDO: Bloco de carregar o LSTM ---
        
        scaler_minmax = joblib.load(io.BytesIO(_load_artifact_from_blob(SCALER_MINMAX_PATH)))
        scaler_std = joblib.load(io.BytesIO(_load_artifact_from_blob(SCALER_STD_PATH)))
        pca_obj = joblib.load(io.BytesIO(_load_artifact_from_blob(PCA_PATH))) 

        # --- Processar dados de entrada ---
        X_unseen_scaled_minmax, X_unseen_pca, X_unseen_scaled_std = model_utils.preprocess_unseen_data(
            X_unseen, scaler_minmax, scaler_std, pca_obj
        )

        # --- Fazer predições com os 6 modelos ---
        predictions_scaled = model_scaled.predict(X_unseen_scaled_minmax)
        predictions_pca = model_pca.predict(X_unseen_pca)
        predictions_ridge = model_ridge.predict(X_unseen_scaled_std) 
        predictions_lasso = model_lasso.predict(X_unseen_scaled_std) 
        
        steps_to_forecast = len(X_unseen)
        predictions_hw = model_hw.forecast(steps=steps_to_forecast)
        predictions_arima = model_arima.forecast(steps=steps_to_forecast) 
        
        # --- REMOVIDO: Bloco de predição do LSTM ---

        # --- Combinar e Proteger Saída (Profa. Thais) ---
        results_df = pd.DataFrame({
            'predicted_Linear_Scaled': predictions_scaled,
            'predicted_Linear_PCA': predictions_pca,
            'predicted_Ridge': predictions_ridge,
            'predicted_Lasso': predictions_lasso,
            'predicted_HoltWinters': predictions_hw,
            'predicted_ARIMA': predictions_arima
            # --- REMOVIDO: 'predicted_LSTM' ---
        })
        
        results_df.to_csv(results_csv_path, index=False)

        secured_data_path, secured_freq_path = security_utils.secure_file(
            results_csv_path, os.path.join(UPLOAD_FOLDER, 'predictions_all_secured'), ENCRYPTION_KEY
        )

        # --- Avaliação (Prof. Fernando) ---
        if has_labels:
            performance_report = {
                "1_Linear_MinMax (PI_Base)": {"R2_Score": r2_score(y_true, predictions_scaled)},
                "2_Linear_PCA (Thais)": {"R2_Score": r2_score(y_true, predictions_pca)},
                "3.Ridge_L2 (Fernando)": {"R2_Score": r2_score(y_true, predictions_ridge)},
                "4.Lasso_L1 (Fernando)": {"R2_Score": r2_score(y_true, predictions_lasso)},
                "5_HoltWinters (Miro)": {"R2_Score": r2_score(y_true, predictions_hw)},
                "6_ARIMA (Miro)": {"R2_Score": r2_score(y_true, predictions_arima)}
                # --- REMOVIDO: "7_LSTM (Miro)" ---
            }
        else:
            performance_report = "Sem rótulos, avaliação não realizada."

        return {
            "status": "success",
            "resultado_compactado_path": secured_data_path,
            "tabela_de_frequencias_path": secured_freq_path,
            "performance": performance_report,
            "download_links": {
                 "secured_predictions_data": f"/download?file={os.path.basename(secured_data_path)}",
                 "secured_predictions_frequency_table": f"/download?file={os.path.basename(secured_freq_path)}"
            }
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")
    finally:
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
        if os.path.exists(results_csv_path):
            os.remove(results_csv_path)

@app.get("/download", tags=["Modelo"])
async def download(file: str = Query(..., description="O nome do arquivo para baixar (ex: predictions_all_secured.huff)")):
    """
    Baixa um arquivo de predição protegido (.huff ou .freq.json).
    """
    file_path = os.path.join(UPLOAD_FOLDER, file)
    
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='application/octet-stream', filename=file)
    else:
        raise HTTPException(status_code=404, detail="Arquivo não encontrado ou já baixado.")

@app.post("/reset", tags=["Modelo"])
async def reset_model():
    """
    Reseta o modelo removendo os 9 artefatos do Azure Blob Storage.
    """
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
        
        deleted_count = 0
        if container_client.exists():
            blob_list = container_client.list_blobs()
            for blob in blob_list:
                if blob.name in ALL_ARTIFACT_PATHS:
                    container_client.delete_blob(blob.name)
                    deleted_count += 1
        
        return {"message": f"{deleted_count} artefatos resetados do Azure Blob Storage."}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao resetar: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)