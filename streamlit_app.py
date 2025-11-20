# ============================================================
# STREAMLIT APP — Versão Final (Persistência de Estado)
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
API_URL = "http://pi-fastapi-backend-eac6a6dfayh0hxgy.canadacentral-01.azurewebsites.net"

# -------------------------
# FUNÇÃO DE LIMPEZA DE NOMES
# -------------------------
def format_model_name(technical_name):
    """Transforma nomes técnicos feios em nomes comerciais bonitos."""
    name = technical_name.replace("predicted_", "")
    mapping = {
        "1_Linear_MinMax": "Linear (MinMax)",
        "2_Linear_PCA": "Linear (PCA)",
        "3.Ridge_L2": "Ridge Regression",
        "3_Ridge_L2": "Ridge Regression",
        "4.Lasso_L1": "Lasso Regression",
        "4_Lasso_L1": "Lasso Regression",
        "5_HoltWinters": "Holt-Winters",
        "6_ARIMA": "ARIMA"
    }
    if name in mapping: return mapping[name]
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
O **Projeto Integrador (PI)** foi desenvolvido com o propósito de unificar os conteúdos apresentados ao longo do semestre nas quatro disciplinas centrais:
* **Computação em Nuvem** (Denis)
* **Aprendizado Supervisionado** (Fernando)
* **Séries Temporais** (Miro)
* **Transformação, Compactação e Redução de Dados** (Thais)
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
            <li>Envio para API FastAPI.</li>
            <li>Treino de <b>6 Modelos</b> na nuvem.</li>
            <li>Persistência no <b>Azure Blob Storage</b>.</li>
            <li>Cálculo de <b>R² Esperado</b> (Validação Temporal).</li>
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
                    with st.spinner("Treinando modelos na nuvem..."):
                        try:
                            files = {"file": train_file}
                            response = requests.post(f"{API_URL}/train", files=files)
                            result = response.json()

                            if response.status_code == 200:
                                st.toast("Treino concluído!", icon="✅")
                                
                                # --- PROCESSAMENTO DOS DADOS ---
                                perf_dict = result["expected_performance_R2_comparison"]
                                clean_perf_dict = {format_model_name(k): v for k, v in perf_dict.items()}
                                perf_df = pd.DataFrame.from_dict(clean_perf_dict, orient='index', columns=['R2'])
                                perf_df = perf_df.sort_values(by="R2", ascending=False)
                                
                                best_model = perf_df['R2'].idxmax()
                                best_r2 = perf_df['R2'].max()
                                
                                st.markdown("### 🏆 Resultado do Treinamento")
                                
                                # --- LAYOUT 3 COLUNAS ---
                                c1, c2, c3 = st.columns([1.2, 1.5, 2])
                                
                                with c1:
                                    st.markdown(f"""
                                    <div class='metric-card'>
                                        <div class='metric-label'>Melhor Modelo</div>
                                        <div class='metric-value'>{best_model}</div>
                                        <div class='metric-desc'>{interpret_r2(best_r2)}<br>(R²: {best_r2:.4f})</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with c2:
                                    st.caption("Ranking Completo")
                                    st.dataframe(perf_df, use_container_width=True, height=250)
                                
                                with c3:
                                    st.caption("Comparativo Visual")
                                    perf_df_chart = perf_df.reset_index().rename(columns={'index': 'Modelo'})
                                    chart = alt.Chart(perf_df_chart).mark_bar().encode(
                                        x=alt.X('R2', scale=alt.Scale(zero=False), title="R² Score"),
                                        y=alt.Y('Modelo', sort='-x', title=""),
                                        color=alt.Color('R2', scale=alt.Scale(scheme='blues'), legend=None),
                                        tooltip=['Modelo', 'R2']
                                    ).properties(height=250)
                                    st.altair_chart(chart, use_container_width=True)
                                
                                st.session_state.history.append({"Hora": datetime.now().strftime("%H:%M"), "Ação": "Treino", "Status": "Sucesso"})
                            else:
                                st.error(f"Erro: {result.get('detail')}")
                        except Exception as e:
                            st.error(f"Erro: {e}")
            else:
                st.error("Colunas incorretas.")
        except:
            st.error("Erro ao ler CSV.")

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
            <li>Download dos modelos do <b>Blob Storage</b>.</li>
            <li><b>Com Rótulos:</b> Gráfico Comparativo (Real vs Previsto).</li>
            <li><b>Sem Rótulos:</b> Geração de arquivo.</li>
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
            with st.spinner("Calculando..."):
                try:
                    test_file.seek(0)
                    files = {"file": test_file}
                    response = requests.post(f"{API_URL}/predict", files=files)
                    
                    if response.status_code == 200:
                        # --- AQUI ESTÁ A CORREÇÃO: SALVAR NA SESSÃO ---
                        st.session_state.test_results = response.json()
                        st.toast("Predição concluída!", icon="✅")
                        st.session_state.history.append({"Hora": datetime.now().strftime("%H:%M"), "Ação": "Predição", "Status": "Sucesso"})
                    else:
                        st.error("Erro na API.")
                except Exception as e:
                    st.error(f"Erro crítico: {e}")

        # --- RENDERIZAÇÃO (FORA DO IF DO BOTÃO) ---
        # Se existirem resultados na memória, mostre-os (independente do botão ter sido clicado agora)
        if st.session_state.test_results is not None:
            res = st.session_state.test_results
            
            links = res.get("download_links", {})
            csv_url = f"{API_URL}{links.get('csv')}"

            st.markdown("### 📦 Downloads")
            st.markdown(f"""
            <div>
                <a href="{csv_url}" class="download-btn" target="_blank">📄 CSV</a>
                <a href="{API_URL}{links.get('huff')}" class="download-btn" target="_blank">🔒 .huff</a>
                <a href="{API_URL}{links.get('freq')}" class="download-btn" target="_blank">🔑 Key</a>
            </div>
            """, unsafe_allow_html=True)

            perf_real = res.get("performance")
            
            try:
                df_results = pd.read_csv(csv_url)
                if 'true_time' in df_results.columns:
                    st.markdown("### 📊 Real vs Previsto")
                    
                    # Preparação dos dados
                    pred_cols = [c for c in df_results.columns if c.startswith('predicted_')]
                    col_map = {c: format_model_name(c) for c in pred_cols}
                    reverse_map = {v: k for k, v in col_map.items()}
                    
                    default_ix = 0
                    if isinstance(perf_real, dict):
                        perf_flat = {k: v["R2_Score"] for k, v in perf_real.items()}
                        best_tech_name = max(perf_flat, key=perf_flat.get)
                        for i, col in enumerate(pred_cols):
                            if best_tech_name.split('_')[1] in col:
                                default_ix = i
                                break

                    # O Selectbox agora não reinicia a tela inteira
                    selected_nice_name = st.selectbox(
                        "Escolha o modelo:", 
                        list(col_map.values()), 
                        index=default_ix
                    )
                    selected_col = reverse_map[selected_nice_name]
                    
                    # Gráfico de Linha
                    chart_data = pd.DataFrame({
                        'Índice': range(len(df_results)),
                        'Real': df_results['true_time'],
                        f'Previsto ({selected_nice_name})': df_results[selected_col]
                    })
                    chart_long = chart_data.melt('Índice', var_name='Legenda', value_name='Valor')
                    
                    line_chart = alt.Chart(chart_long).mark_line(point=True).encode(
                        x='Índice',
                        y=alt.Y('Valor', scale=alt.Scale(zero=False)),
                        color=alt.Color('Legenda', scale=alt.Scale(domain=['Real', f'Previsto ({selected_nice_name})'], range=['#10B981', '#3B82F6'])),
                        tooltip=['Índice', 'Legenda', 'Valor']
                    ).properties(height=350).interactive()
                    st.altair_chart(line_chart, use_container_width=True)

                    # Ranking de Barras (Teste)
                    if isinstance(perf_real, dict):
                        clean_perf_real = {format_model_name(k): v["R2_Score"] for k, v in perf_real.items()}
                        df_perf = pd.DataFrame.from_dict(clean_perf_real, orient='index', columns=['R2'])
                        df_chart = df_perf.reset_index().rename(columns={'index': 'Modelo'})
                        
                        bar_chart = alt.Chart(df_chart).mark_bar().encode(
                            x=alt.X('R2', scale=alt.Scale(zero=False)),
                            y=alt.Y('Modelo', sort='-x', title=""),
                            color=alt.Color('R2', scale=alt.Scale(scheme='reds')),
                            tooltip=['Modelo', 'R2']
                        ).properties(height=200)
                        st.altair_chart(bar_chart, use_container_width=True)
                else:
                    st.warning("⚠️ Gráfico indisponível (Arquivo sem gabarito).")
            except Exception as e:
                st.error(f"Erro visualização: {e}")

if st.checkbox("Histórico"):
    st.table(st.session_state.history)
