# 🧠 Laboratório de Modelos de Regressão e Séries Temporais

Este repositório contém a solução desenvolvida para o Projeto Integrador do curso de Ciência de Dados e Inteligência Artificial. O sistema é uma plataforma *end-to-end* para treinamento, seleção automática do melhor modelo e persistência em nuvem, com foco em segurança, robustez estatística e eficiência.

## 🎯 Objetivo do Projeto

O objetivo foi desenvolver uma aplicação capaz de prever eventos em séries temporais (coluna `time`) com base em 5 observações anteriores. O projeto integra conceitos de quatro disciplinas fundamentais:

* **Computação em Nuvem:** Treinamento remoto, persistência de artefatos (Azure Blob Storage) e API RESTful.
* **Aprendizado Supervisionado:** Comparação de modelos de regressão (Linear, Ridge, Lasso, ElasticNet).
* **Séries Temporais:** Aplicação de modelos específicos (Holt-Winters, ARIMA) e validação cruzada temporal (`TimeSeriesSplit`).
* **Transformação e Compactação de Dados:** Pipeline de pré-processamento estatístico e implementação de segurança (Huffman + Cifra XOR).

## 🚀 Funcionalidades Principais

## 🚀 Funcionalidades Principais

### 1. Treinamento Inteligente em Nuvem
* **Upload** de arquivos `.csv` para treino.
* **Pipeline de Pré-processamento Robusto:**
    * **Padronização Estatística (`StandardScaler`):** * **IMPORTANTE:** O Scaler é ajustado (`fit`) **apenas** nos dados de treino para aprender a média e desvio padrão. 
        * Durante o teste ou produção, utilizamos esses mesmos parâmetros para apenas transformar (`transform`) os novos dados. Isso garante que não haja *Data Leakage* (contaminação pelo futuro).
    * **Validação Cruzada Temporal:** Utilização de `TimeSeriesSplit` para respeitar a ordem cronológica dos dados durante a validação.
* **Treinamento Simultâneo de 4 Modelos Lineares:**
    1.  Regressão Linear (Standard)
    2.  Ridge Regression (Regularização L2)
    3.  Lasso Regression (Regularização L1)
    4.  Elastic Net (Híbrido L1+L2)
* **Seleção Automática (MCDA):** O sistema avalia os modelos e elege o "Vencedor" baseado em um ranking multicritério (R², RMSE e MAE).

### 2. Teste e Aplicação (*Best Model Strategy*)
* **Eficiência Computacional:** Ao receber uma nova base de teste, o sistema carrega e executa **apenas o modelo vencedor**.
* **Segurança e Compactação:** Os arquivos de saída são entregues criptografados e compactados (Huffman + XOR).

## 📂 Estrutura de Arquivos
* **`app/main.py`**: API RESTful e gerenciamento de modelos.
* **`app/model_utils.py`**: Pipeline de Ciência de Dados (Scaler e Modelos scikit-learn).
* **`app/security_utils.py`**: Algoritmos de segurança (Huffman + XOR).
* **`requirements.txt`**: Dependências limpas (sem bibliotecas pesadas desnecessárias).