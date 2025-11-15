import streamlit as st
import requests
import pandas as pd
import json

# URL do seu "motor" (o main.py)
API_URL = "http://127.0.0.1:8080"

# --- Configuração da Página ---
st.set_page_config(
    page_title="PI - Laboratório de Modelos",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Laboratório de Modelos Acadêmicos (PI)")
st.markdown("Interface de usuário para o PI, integrando as matérias dos Profs. Miro, Fernando, Thais e Denis.")

# --- Colunas para o Layout ---
col1, col2 = st.columns(2)

# --- Coluna 1: Ações (Train, Predict) ---
with col1:
    
    # --- Card de Treinamento ---
    with st.container(border=True):
        st.header("1. Treinar Modelos")
        st.markdown("Envie seu `PI_train.csv` para treinar os 6 modelos.")
        
        train_file = st.file_uploader("Selecione o arquivo de TREINO (.csv)", key="train")
        
        if st.button("Executar Treinamento", type="primary"):
            if train_file:
                with st.spinner("Treinando 6 modelos... (Isso pode levar alguns minutos)..."):
                    files = {'file': (train_file.name, train_file, 'text/csv')}
                    
                    try:
                        response = requests.post(f"{API_URL}/train", files=files)
                        
                        if response.status_code == 200:
                            st.session_state.train_results = response.json()
                            st.session_state.last_error = None
                        else:
                            st.session_state.train_results = None
                            st.session_state.last_error = response.json().get('detail', 'Erro desconhecido')
                            
                    except requests.exceptions.ConnectionError:
                        st.session_state.last_error = "ERRO DE CONEXÃO: O backend (main.py) está rodando?"
                    except Exception as e:
                        st.session_state.last_error = f"Erro inesperado: {e}"
            else:
                st.warning("Por favor, selecione um arquivo de treino.")

    # --- Card de Predição ---
    with st.container(border=True):
        st.header("2. Fazer Predição")
        st.markdown("Envie seu `PI_test.csv` para gerar predições e o relatório de performance.")
        
        predict_file = st.file_uploader("Selecione o arquivo de TESTE (.csv)", key="predict")
        
        if st.button("Executar Predição"):
            if predict_file:
                with st.spinner("Realizando predições nos 6 modelos..."):
                    files = {'file': (predict_file.name, predict_file, 'text/csv')}
                    
                    try:
                        response = requests.post(f"{API_URL}/predict", files=files)
                        
                        if response.status_code == 200:
                            st.session_state.predict_results = response.json()
                            st.session_state.last_error = None
                        else:
                            st.session_state.predict_results = None
                            st.session_state.last_error = response.json().get('detail', 'Erro desconhecido')

                    except requests.exceptions.ConnectionError:
                        st.session_state.last_error = "ERRO DE CONEXÃO: O backend (main.py) está rodando?"
                    except Exception as e:
                        st.session_state.last_error = f"Erro inesperado: {e}"
            else:
                st.warning("Por favor, selecione um arquivo de predição.")

    # --- Card de Reset ---
    with st.container(border=True):
        st.header("3. Resetar Modelos")
        st.markdown("Apaga todos os modelos treinados na nuvem (Azure Blob Storage).")
        
        if st.button("Resetar Modelos na Nuvem"):
            with st.spinner("Resetando..."):
                try:
                    response = requests.post(f"{API_URL}/reset")
                    if response.status_code == 200:
                        st.session_state.last_error = None
                        st.success(response.json().get("message", "Reset concluído!"))
                    else:
                        st.session_state.last_error = response.json().get('detail', 'Erro ao resetar')
                except requests.exceptions.ConnectionError:
                     st.session_state.last_error = "ERRO DE CONEXÃO: O backend (main.py) está rodando?"


# --- Coluna 2: Resultados ---
with col2:
    st.header("Resultados da Operação")

    # Mostrar o último erro que aconteceu
    if 'last_error' in st.session_state and st.session_state.last_error:
        st.error(st.session_state.last_error)
        st.session_state.last_error = None # Limpa o erro

    # Mostrar resultados do Treino
    if 'train_results' in st.session_state and st.session_state.train_results:
        st.success(st.session_state.train_results.get("message"))
        st.subheader("Expectativa de Performance (R²)")
        
        perf_data = st.session_state.train_results.get("expected_performance_R2_comparison (Validação)")
        if perf_data:
            perf_df = pd.DataFrame.from_dict(perf_data, orient='index', columns=['R²'])
            
            # --- CORREÇÃO AQUI ---
            st.bar_chart(perf_df)
            # --- FIM DA CORREÇÃO ---
        
        with st.expander("Ver JSON de Resposta (Treino)"):
            st.json(st.session_state.train_results)
        
        st.session_state.train_results = None # Limpa o resultado

    # Mostrar resultados da Predição
    if 'predict_results' in st.session_state and st.session_state.predict_results:
        st.success(st.session_state.predict_results.get("status", "Predição concluída!"))
        st.subheader("Performance Real (R²)")
        
        perf_data = st.session_state.predict_results.get("performance")
        if isinstance(perf_data, dict):
            perf_df = pd.DataFrame.from_dict(perf_data, orient='index')
            
            # --- CORREÇÃO AQUI ---
            st.bar_chart(perf_df)
            # --- FIM DA CORREÇÃO ---
            
        else:
            st.info(perf_data) # Mostra "Sem rótulos, avaliação não realizada."
        
        st.subheader("Download dos Resultados Seguros")
        st.markdown("Lembre-se: Você precisa de ambos os arquivos (Huffman da Profa. Thais).")
        
        links = st.session_state.predict_results.get("download_links", {})
        if links:
            st.markdown(f"**1. Arquivo de Dados (.huff):** `{links.get('secured_predictions_data')}`")
            st.markdown(f"**2. Tabela de Frequência (.freq.json):** `{links.get('secured_predictions_frequency_table')}`")
            st.markdown(f"_(Copie e cole o link no seu navegador para baixar: `{API_URL}/download?file=...`)_")

        with st.expander("Ver JSON de Resposta (Predição)"):
            st.json(st.session_state.predict_results)
            
        st.session_state.predict_results = None # Limpa o resultado