🧠 Laboratório de Modelos de Regressão e Séries Temporais
Este repositório contém a solução desenvolvida para o Projeto Integrador do curso de Ciência de Dados e Inteligência Artificial da PUC-Campinas. O sistema é uma plataforma end-to-end para treinamento, persistência em nuvem e avaliação de modelos de regressão e séries temporais, com foco em segurança e compactação de dados.

🎯 Objetivo do Projeto
O objetivo foi desenvolver uma aplicação capaz de prever eventos em séries temporais (coluna time) com base em 5 observações anteriores. O projeto integra conceitos de quatro disciplinas fundamentais:

Computação em Nuvem: Treinamento remoto e persistência de modelos (Azure Blob Storage).

Aprendizado Supervisionado: Comparação de modelos de regressão (Linear, Ridge, Lasso).

Séries Temporais: Aplicação de modelos específicos (Holt-Winters, ARIMA).

Transformação e Compactação de Dados: Pipeline de pré-processamento e implementação manual de criptografia/compactação (Huffman + Cifra XOR).

🚀 Funcionalidades Principais
1. Treinamento em Nuvem
Upload de arquivos .csv para treino.

Pipeline automático de pré-processamento:

Normalização (MinMax e StandardScaler).

Redução de dimensionalidade (PCA - 90% de variância).

Treinamento simultâneo de 6 modelos:

Regressão Linear (Dados Normalizados)

Regressão Linear (com PCA)

Ridge Regression (L2)

Lasso Regression (L1)

Holt-Winters (Suavização Exponencial)

ARIMA

Persistência dos artefatos (modelos e scalers) no Azure Blob Storage.

2. Teste e Predição
Upload de base de teste (com ou sem rótulos).

Geração de previsões utilizando os modelos salvos na nuvem.

Avaliação Automática: Se a base contiver rótulos, o sistema calcula o R² Score e plota gráficos comparativos (Real vs Previsto).

3. Segurança e Compactação (Ponta-a-Ponta)
Implementação de um algoritmo híbrido de segurança:

Compactação: Codificação de Huffman (baseada na frequência de caracteres).

Criptografia: Cifra XOR aplicada sobre os dados binários compactados.

Os arquivos de saída (resultados) são entregues ao usuário neste formato seguro (.huff), garantindo a integridade e confidencialidade no transporte.

🛠️ Arquitetura da Solução
A solução foi dividida em dois componentes principais (Frontend e Backend) seguindo a arquitetura de microsserviços:

Snippet de código

graph LR
A[Usuário / Streamlit] -- Upload CSV --> B(FastAPI Backend)
B -- Processamento ML --> C{Model Utils}
C -- Salvar/Carregar --> D[(Azure Blob Storage)]
B -- Segurança (Huffman+XOR) --> E{Security Utils}
E -- Download (.huff) --> A
Estrutura de Arquivos
main.py: API RESTful construída com FastAPI. Gerencia rotas, treinamento e conexão com a Azure.

streamlit_app.py: Interface visual interativa construída com Streamlit.

model_utils.py: Módulo contendo a lógica de Data Science (Split, Scalers, PCA, Treino de Sklearn/Statsmodels).

security_utils.py: Implementação customizada da compressão Huffman e criptografia XOR.

requirements.txt: Dependências do projeto.

⚙️ Instalação e Execução
Pré-requisitos
Python 3.9 ou superior.

Conta no Microsoft Azure (ou emulador Azurite local).

1. Clonar o repositório
Bash

git clone https://github.com/seu-usuario/nome-do-repo.git
cd nome-do-repo
2. Instalar dependências
Bash

pip install -r requirements.txt
3. Configurar Variáveis de Ambiente
Crie um arquivo .env ou exporte a variável de conexão com o Azure Storage:

Bash

# Exemplo para Linux/Mac
export AZURE_STORAGE_CONNECTION_STRING="sua_connection_string_aqui"

# Exemplo para Windows (PowerShell)
$env:AZURE_STORAGE_CONNECTION_STRING="sua_connection_string_aqui"
Nota: Se a variável não for definida, o sistema tentará conectar no emulador local (Azurite).

4. Executar a Aplicação
O sistema requer que o Backend e o Frontend rodem simultaneamente. Abra dois terminais:

Terminal 1 (Backend API):

Bash

uvicorn main:app --reload --port 8000
Terminal 2 (Frontend Streamlit):

Bash

streamlit run streamlit_app.py
Acesse a aplicação em: http://localhost:8501

📊 Guia de Uso
Painel de Controle: A barra lateral mostra o status da conexão com a API e Azure. Use o botão "Resetar sistema" para limpar modelos antigos da nuvem.

Treinamento:

Faça upload do arquivo train.csv.

Clique em "Iniciar Treinamento".

Analise o gráfico de ranking dos melhores modelos baseado no R² esperado.

Predição:

Faça upload do arquivo test.csv.

O sistema baixará os modelos da nuvem e gerará as previsões.

Download: Baixe os resultados em formato CSV ou formato seguro (.huff + chave).

Visualização: Se o arquivo tiver gabarito, um gráfico interativo comparará a curva Real vs Prevista.

👥 Autores
Projeto desenvolvido pelos alunos de Ciência de Dados e IA (PUC-Campinas):

Matheus Gomes (RA: 23004938)

Maria Eduarda S. A. P. Costa (RA: 23005493)

📝 Licença
Este projeto está sob a licença MIT. Consulte o arquivo LICENSE para mais detalhes.
