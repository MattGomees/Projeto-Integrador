# 🧠 Laboratório de Modelos de Regressão e Séries Temporais

Este repositório contém a solução desenvolvida para o Projeto Integrador do curso de Ciência de Dados e Inteligência Artificial. O sistema é uma plataforma *end-to-end* para treinamento, seleção automática do melhor modelo de regressão e persistência em nuvem, com foco em segurança, robustez estatística e eficiência.

## 🎯 Objetivo do Projeto

Desenvolver uma aplicação capaz de prever eventos em séries temporais com base em 5 observações anteriores. O projeto integra conceitos de:

* **Computação em Nuvem:** API RESTful com FastAPI e persistência de artefatos no Azure Blob Storage.
* **Aprendizado Supervisionado:** Comparação e seleção automática entre modelos de regressão (Linear, Ridge, Lasso, ElasticNet).
* **Séries Temporais:** Validação cruzada temporal (`TimeSeriesSplit`) e prevenção de *Data Leakage*.
* **Segurança:** Compactação e criptografia dos resultados (Huffman + Cifra XOR).

## 🚀 Funcionalidades Principais

1.  **Treinamento Automatizado (AutoML):**
    * Treina simultaneamente 4 algoritmos lineares.
    * Avalia modelos usando R², RMSE e MAE.
    * Seleciona automaticamente o "modelo vencedor" para produção.
    * Salva o scaler e o modelo treinado na nuvem.

2.  **Predição Segura:**
    * Aplica apenas o modelo vencedor nos novos dados.
    * Gera arquivos de saída criptografados (`.huff`) para garantir a confidencialidade das predições.

3.  **Interface Gráfica:**
    * Dashboard interativo em Streamlit para fácil operação por usuários não técnicos.

## 📋 Pré-requisitos dos Dados

Para que o treinamento e a predição funcionem corretamente, o arquivo `.csv` deve seguir estritamente o formato de janelas de tempo (*lags*):

| Colunas Obrigatórias | Descrição |
| :--- | :--- |
| `time-5`, `time-4`, `time-3`, `time-2`, `time-1` | As 5 observações passadas (Features). A ordem é importante. |
| `time` | O valor alvo (Target). Obrigatório para treino; opcional para predição. |

## 🛠️ Como Executar

### Opção 1: Via Docker (Recomendado)

O projeto está containerizado para facilitar a execução.

1.  **Construir a imagem:**
    ```bash
    docker build -t projeto-integrador .
    ```

2.  **Rodar o container:**
    ```bash
    docker run -p 8000:8000 projeto-integrador
    ```
    *A API estará disponível em `http://localhost:8000`.*

### Opção 2: Execução Local

Certifique-se de ter o Python 3.10+ instalado.

1.  **Instalar dependências:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Iniciar o Backend (API):**
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```

3.  **Iniciar o Frontend (Streamlit):**
    Em um novo terminal, execute:
    ```bash
    streamlit run streamlit_app.py
    ```

## 🔌 Documentação da API

Após iniciar a aplicação (localmente ou via Docker), a documentação interativa (Swagger UI) estará disponível em:

* **URL:** `http://localhost:8000/dev-docs`

### Endpoints Principais
* `POST /train`: Recebe CSV de treino, executa o pipeline e salva o melhor modelo.
* `POST /predict`: Recebe CSV de teste e retorna as predições criptografadas.
* `GET /download`: Permite baixar os arquivos gerados (CSV, .huff, .json).
* `POST /reset`: Reseta o sistema, apagando modelos salvos na nuvem.

## 👥 Autores

* **Matheus Gomes** (Email: matheus.rg@puccampinas.edu.br)
* **Maria Eduarda S. A. P. Costa** (Email: maria.esapc@puccampinas.edu.br)