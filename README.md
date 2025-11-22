# 🧠 Laboratório de Modelos de Regressão e Séries Temporais

Este repositório contém a solução desenvolvida para o Projeto Integrador do curso de Ciência de Dados e Inteligência Artificial. O sistema é uma plataforma *end-to-end* para treinamento, seleção automática do melhor modelo e persistência em nuvem, com foco em segurança, robustez estatística e eficiência.

## 🎯 Objetivo do Projeto

O objetivo foi desenvolver uma aplicação capaz de prever eventos em séries temporais (coluna `time`) com base em 5 observações anteriores. O projeto integra conceitos de quatro disciplinas fundamentais:

* **Computação em Nuvem:** Treinamento remoto, persistência de artefatos (Azure Blob Storage) e API RESTful.
* **Aprendizado Supervisionado:** Comparação de modelos de regressão (Linear, Ridge, Lasso, ElasticNet).
* **Séries Temporais:** Aplicação de modelos específicos (Holt-Winters, ARIMA) e validação cruzada temporal (`TimeSeriesSplit`).
* **Transformação e Compactação de Dados:** Pipeline de pré-processamento estatístico e implementação de segurança (Huffman + Cifra XOR).

## 🚀 Funcionalidades Principais

### 1. Treinamento Inteligente em Nuvem
* **Upload** de arquivos `.csv` para treino.
* **Pipeline de Pré-processamento Robusto:**
    * **Padronização Estatística (`StandardScaler`):** Aplica transformação Z-score para alinhar a distribuição dos dados, garantindo convergência ótima para modelos lineares.
    * **Prevenção de *Data Leakage*:** Durante a validação, os parâmetros de escala são ajustados exclusivamente dentro de cada "janela" de treino, simulando um cenário real de previsão.
* **Treinamento Simultâneo de 6 Modelos:**
    1.  Regressão Linear (Standard)
    2.  Ridge Regression (Regularização L2)
    3.  Lasso Regression (Regularização L1)
    4.  Elastic Net (Híbrido L1+L2)
    5.  Holt-Winters (Suavização Exponencial)
    6.  ARIMA (AutoRegressive Integrated Moving Average)
* **Seleção Automática (MCDA):** O sistema avalia os modelos via validação cruzada temporal e elege o "Vencedor" baseado em um ranking multicritério (Soma dos ranks de R², RMSE e MAE).
* **Persistência Otimizada:** Apenas os artefatos necessários e a identificação do modelo vencedor são gerenciados no Azure Blob Storage.

### 2. Teste e Aplicação (*Best Model Strategy*)
* **Eficiência Computacional:** Ao receber uma nova base de teste, o sistema carrega e executa **apenas o modelo vencedor** definido na etapa de treino. Isso reduz a latência e o consumo de memória.
* **Avaliação Automática:** Se a base de teste contiver rótulos (gabarito), o sistema calcula as métricas de desempenho (R², RMSE, MAE) exclusivamente para o modelo campeão.

### 3. Segurança e Compactação (Ponta-a-Ponta)
Implementação de um algoritmo híbrido de segurança nos arquivos de saída:
* **Compactação:** Codificação de Huffman (baseada na frequência de caracteres do arquivo).
* **Criptografia:** Cifra XOR aplicada sobre os dados binários compactados.

Os arquivos de saída são entregues ao usuário neste formato seguro (`.huff`), garantindo a integridade e confidencialidade no transporte.

## 📂 Estrutura de Arquivos

Abaixo está a descrição dos principais arquivos e diretórios do projeto:

* **`app/main.py`**: API RESTful. Gerencia o ciclo de vida do treino, conexão com Azure e predição seletiva.
* **`streamlit_app.py`**: Interface visual. Exibe os resultados e gráficos do modelo campeão.
* **`app/model_utils.py`**: Lógica de Data Science (Pipeline de treino, Validação Temporal Rigorosa, StandardScaler).
* **`app/security_utils.py`**: Implementação da compressão Huffman e criptografia XOR.
* **`requirements.txt`**: Dependências do projeto.

---

## ⚙️ Instalação e Execução

### Pré-requisitos

* **Python 3.9** ou superior.
* Conta no **Microsoft Azure** (ou emulador Azurite local).

### 1. Instalar dependências

Execute o seguinte comando no terminal para instalar as bibliotecas necessárias:

```bash
pip install -r requirements.txt