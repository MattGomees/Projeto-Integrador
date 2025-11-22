# ============================================================
# STREAMLIT APP — Versão Final (Refinada sem PCA/MinMax)
# ============================================================
import streamlit as st
import requests
import pandas as pd
import altair as alt
from datetime import datetime
import os

# -------------------------
# CONFIGURAÇÃO DA PÁGINA
# -------------------------
st.set_page_config(
    page_title="Projeto Integrador",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL da API (Backend)
# IMPORTANTE: Se estiver rodando localmente o backend, mude para "http://localhost:8000"
API_URL = "https://pi-fastapi-backend-eac6a6dfayh0hxgy.canadacentral-01.azurewebsites.net"
# API_URL = "http://localhost:8000" # Descomente para teste local

# -------------------------
# FUNÇÃO DE LIMPEZA DE NOMES
# -------------------------
def format_model_name(technical_name):
    """Transforma nomes técnicos novos em nomes comerciais bonitos."""
    
    # Remove o prefixo 'predicted_' se houver
    name = technical_name.replace("predicted_", "")
    
    mapping = {
        "1_Linear_Std": "Linear Regression (Standard)",
        "3_Ridge_L2": "Ridge Regression",
        "4_Lasso_L1": "Lasso Regression",
        "5_ElasticNet": "Elastic Net", 
        "6_HoltWinters": "Holt-Winters",
        "7_ARIMA": "ARIMA"
    }
    
    if name in mapping: return mapping[name]
    # Busca por parte do nome para segurança
    for key, nice_name in mapping.items():
        if key in name: return nice_name
    return name

def interpret_r2(value):
    if value > 0.85: return "Excelente 🌟"
    elif value > 0.70: return "Muito Bom ✅"
    elif value > 0.50: return "Regular ⚠️"
    else: return "Fraco 🔻"

# -------------------------
# ESTILO CUSTOMIZADO (CSS)
# -------------------------
st.markdown("""
    <style>
        /* Fundo e Fontes */
        .stApp { background-color: #0F172A; color: #F1F5F9; }
        
        /* Cards Explicativos */
        .info-card { background-color: #1E293B; border-left: 4px solid #3B82F6; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
        .info-card h4 { color: #60A5FA; margin-top: 0; font-size: 1.1rem; }
        .info-card p, .info-card li { font-size: 0.9rem; color: #CBD5E1; line-height: 1.5; }

        /* Cards de Métricas (Pódio) */
        .metric-card { background: #1E293B; padding: 20px; border-radius: 10px; border: 1px solid #334155; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center; }
        .metric-value { font-size: 2.2rem; font-weight: 700; color: #38BDF8; word-wrap: break-word; }
        .metric-label { color: #94A3B8; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; }
        .metric-desc { font-size: 1rem; color: #10B981; margin-top: 10px; font-weight: 500; }

        /* Botões de Download */
        .download-btn { display: inline-block; background-color: #1E293B; color: #60A5FA; padding: 10px 20px; border-radius: 8px; border: 1px solid #3B82F6; text-decoration: none; margin-right: 10px; font-weight: 600; transition: all 0.3s ease; }
        .download-btn:hover { background-color: #3B82F6; color: white; }

        /* Status na Sidebar (Sem Emojis, Limpo) */
        .status-item { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.85rem; color: #94A3B8; border-bottom: 1px solid #1E293B; padding-bottom: 4px; }
        .status-val { color: #CBD5E1; font-weight: 600; }

        /* BOTÃO RESET VERMELHO (Sidebar) */
        div[data-testid="stSidebar"] button {
            border: 1px solid #EF4444 !important;
            color: #EF4444 !important;
            background-color: transparent !important;
            width: 100%;
        }
        div[data-testid="stSidebar"] button:hover {
            background-color: #EF4444 !important;
            color: white !important;
            border: 1px solid #EF4444 !important;
        }
        
    </style>
""", unsafe_allow_html=True)

# Inicializa histórico na sessão
if "history" not in st.session_state:
    st.session_state.history = []

# Inicializa estado dos resultados do teste
if "test_results" not in st.session_state:
    st.session_state.test_results = None

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.title("Painel de Controle")
    
    st.markdown("### Alunos")
    st.markdown("**Matheus Gomes**\nRA: 23004938")
    st.markdown("**Maria Eduarda S. A. P. Costa**\nRA: 23005493")
    
    st.markdown("---")
    st.markdown("### Status do Sistema")
    st.markdown("""
    <div class='status-item'><span>API Backend</span> <span class='status-val'>Online</span></div>
    <div class='status-item'><span>Azure Blob</span> <span class='status-val'>Conectado</span></div>
    <div class='status-item'><span>Segurança</span> <span class='status-val'>Ativa</span></div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    # Botão Vermelho via CSS acima
    if st.button("Resetar sistema"):
        try:
            requests.post(f"{API_URL}/reset")
            st.toast("Sistema resetado!", icon="🧹")
            st.session_state.history = []
            st.session_state.test_results = None # Limpa resultados da memória
        except:
            st.error("API Offline.")

# -------------------------
# HEADER
# -------------------------
st.title("🧠 Projeto Integrador")
st.markdown("""
O **Projeto Integrador (PI)** unifica conceitos de:
* **Computação em Nuvem**
* **Aprendizado Supervisionado**
* **Séries Temporais**
* **Transformação e Compactação de Dados**
""")
st.markdown("---")

# ============================================================
# 1. TREINAMENTO
# ============================================================
st.header("1. Treinamento do Modelo")
col_train_action, col_train_info = st.columns([2, 1])

with col_train_info:
    st.markdown("""
    <div class='info-card'>
        <h4>🔍 Processo de Treino</h4>
        <ul>
            <li>Treino de <b>6 Modelos</b> na nuvem.</li>
            <li>Seleção automática do <b>Melhor Modelo</b>.</li>
            <li>Persistência no <b>Azure Blob Storage</b>.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_train_action:
    train_file = st.file_uploader("📂 Upload da Base de Treino (.csv)", type=["csv"], key="train")
    
    if train_file is not None:
        try:
            df_preview = pd.read_csv(train_file)
            cols_needed = ['time-5', 'time-4', 'time-3', 'time-2', 'time-1', 'time']
            st.markdown("##### 🔎 Data Preview")
            st.dataframe(df_preview.head(), use_container_width=True, height=150)
            
            if set(cols_needed).issubset(df_preview.columns):
                if st.button("🚀 Iniciar Treinamento"):
                    train_file.seek(0)
                    with st.spinner("Treinando e selecionando o melhor modelo..."):
                        try:
                            files = {"file": train_file}
                            response = requests.post(f"{API_URL}/train", files=files)
                            result = response.json()

                            if response.status_code == 200:
                                st.toast("Treino concluído!", icon="✅")
                                
                                # --- PROCESSAMENTO DOS DADOS ---
                                full_metrics_data = result.get("full_expected_performance_metrics", {})
                                
                                # 1. Puxa o nome do vencedor
                                best_model_tech_name = full_metrics_data.pop("best_model_tech_name", None)

                                # 2. Cria o DataFrame
                                clean_perf_dict = {format_model_name(k): v for k, v in full_metrics_data.items()}
                                perf_df = pd.DataFrame.from_dict(clean_perf_dict, orient='index')
                                perf_df.index.name = 'Modelo'
                                
                                # 3. Ordena pelo R2
                                perf_df = perf_df.sort_values(by="R2", ascending=False)
                                
                                best_model_nice = "Desconhecido"
                                best_r2 = 0.0
                                
                                if best_model_tech_name and format_model_name(best_model_tech_name) in perf_df.index:
                                     best_model_nice = format_model_name(best_model_tech_name)
                                     best_r2 = perf_df.loc[best_model_nice, "R2"]
                                else:
                                     best_model_nice = perf_df.index[0]
                                     best_r2 = perf_df.iloc[0]["R2"]
                                
                                st.markdown("### 🏆 Resultado: Melhor Modelo Selecionado")
                                
                                # --- LAYOUT 3 COLUNAS ---
                                c1, c2, c3 = st.columns([1.2, 1.5, 2])
                                
                                with c1:
                                    st.markdown(f"""
                                    <div class='metric-card'>
                                        <div class='metric-label'>Vencedor</div>
                                        <div class='metric-value'>{best_model_nice}</div>
                                        <div class='metric-desc'>{interpret_r2(best_r2)}<br>(R²: {best_r2:.4f})</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with c2:
                                    st.caption("Ranking Completo (R², RMSE, MAE)")
                                    st.dataframe(perf_df.style.format({"R2": "{:.4f}", "RMSE": "{:.4f}", "MAE": "{:.4f}"}), 
                                                 use_container_width=True, height=250) 
                                
                                with c3:
                                    st.caption("Comparativo Visual (R²)")
                                    perf_df_chart = perf_df.reset_index().rename(columns={'Modelo': 'Modelo'})
                                    chart = alt.Chart(perf_df_chart).mark_bar().encode(
                                        x=alt.X('R2', scale=alt.Scale(zero=False), title="R² Score"),
                                        y=alt.Y('Modelo', sort='-x', title=""),
                                        color=alt.Color('R2', scale=alt.Scale(scheme='blues'), legend=None),
                                        tooltip=['Modelo', 'R2', 'RMSE', 'MAE']
                                    ).properties(height=250)
                                    st.altair_chart(chart, use_container_width=True)
                                
                                st.session_state.history.append({"Hora": datetime.now().strftime("%H:%M"), "Ação": "Treino", "Status": f"Melhor: {best_model_nice}"})
                            else:
                                st.error(f"Erro: {result.get('detail')}")
                        except Exception as e:
                            st.error(f"Erro: {e}")
            else:
                st.error("Colunas incorretas.")
        except Exception as e:
             st.error(f"Erro ao ler CSV: {e}")

st.markdown("---")

# ============================================================
# 2. TESTE E APLICAÇÃO (COM PERSISTÊNCIA)
# ============================================================
st.header("2. Teste & Aplicação (Predição)")
col_test_action, col_test_info = st.columns([2, 1])

with col_test_info:
    st.markdown("""
    <div class='info-card'>
        <h4>⚙️ Processo de Teste</h4>
        <ul>
            <li>Usa <b>APENAS</b> o modelo vencedor do treino.</li>
            <li><b>Com Rótulos:</b> Gráfico Comparativo (Real vs Previsto).</li>
            <li>Entrega segura via <b>Huffman + XOR</b>.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_test_action:
    # Callback para limpar resultados anteriores se mudar o arquivo
    def clear_results():
        st.session_state.test_results = None

    test_file = st.file_uploader("📂 Upload da Base de Teste (.csv)", type=["csv"], key="test", on_change=clear_results)

    if test_file is not None:
        # Botão apenas dispara a API e salva na memória
        if st.button("⚡ Gerar Previsões"):
            with st.spinner("Aplicando melhor modelo..."):
                try:
                    test_file.seek(0)
                    files = {"file": test_file}
                    response = requests.post(f"{API_URL}/predict", files=files)
                    
                    if response.status_code == 200:
                        st.session_state.test_results = response.json()
                        st.toast("Predição concluída!", icon="✅")
                        st.session_state.history.append({"Hora": datetime.now().strftime("%H:%M"), "Ação": "Predição", "Status": "Sucesso"})
                    else:
                        st.error(f"Erro na API: {response.json().get('detail', 'Desconhecido')}")
                except Exception as e:
                    st.error(f"Erro crítico: {e}")

        # --- RENDERIZAÇÃO (FORA DO IF DO BOTÃO) ---
        if st.session_state.test_results is not None:
            res = st.session_state.test_results
            
            links = res.get("download_links", {})
            csv_url = f"{API_URL}{links.get('csv')}"
            best_model_used = res.get("best_model_used", "Modelo")
            best_model_nice_name = format_model_name(best_model_used)

            st.markdown("### 📦 Downloads")
            st.markdown(f"""
            <div>
                <a href="{csv_url}" class="download-btn" target="_blank">📄 CSV ({best_model_nice_name})</a>
                <a href="{API_URL}{links.get('huff')}" class="download-btn" target="_blank">🔒 .huff</a>
                <a href="{API_URL}{links.get('freq')}" class="download-btn" target="_blank">🔑 Key</a>
            </div>
            """, unsafe_allow_html=True)

            perf_real = res.get("performance")
            
            try:
                df_results = pd.read_csv(csv_url)
                if 'true_time' in df_results.columns:
                    st.markdown(f"### 📊 Real vs Previsto: {best_model_nice_name}")
                    
                    # Identifica a coluna de predição (agora só existe UMA)
                    pred_cols = [c for c in df_results.columns if c.startswith('predicted_')]
                    
                    if pred_cols:
                        selected_col = pred_cols[0]
                        
                        # Gráfico de Linha Simplificado (Sem Selectbox)
                        chart_data = pd.DataFrame({
                            'Índice': range(len(df_results)),
                            'Real': df_results['true_time'],
                            'Previsto': df_results[selected_col]
                        })
                        chart_long = chart_data.melt('Índice', var_name='Legenda', value_name='Valor')
                        
                        line_chart = alt.Chart(chart_long).mark_line(point=True).encode(
                            x='Índice',
                            y=alt.Y('Valor', scale=alt.Scale(zero=False)),
                            color=alt.Color('Legenda', scale=alt.Scale(domain=['Real', 'Previsto'], range=['#10B981', '#3B82F6'])),
                            tooltip=['Índice', 'Legenda', 'Valor']
                        ).properties(height=350).interactive()
                        st.altair_chart(line_chart, use_container_width=True)

                        # Exibe métricas do modelo único
                        if isinstance(perf_real, dict) and best_model_used in perf_real:
                            metrics = perf_real[best_model_used]
                            st.markdown("##### 📝 Desempenho na Base de Teste")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("R² Score", f"{metrics['R2_Score']:.4f}")
                            c2.metric("RMSE", f"{metrics['RMSE']:.4f}")
                            c3.metric("MAE", f"{metrics['MAE']:.4f}")
                    else:
                        st.warning("Coluna de predição não encontrada no CSV.")

                else:
                    st.info("ℹ️ Arquivo sem gabarito (sem coluna 'time'). Gráfico indisponível.")
            except Exception as e:
                st.error(f"Erro visualização: {e}")

if st.checkbox("Histórico"):
    st.table(st.session_state.history)