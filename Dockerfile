# 1. Imagem Base
FROM python:3.10-slim

# 2. Define o Diretório de Trabalho (PASTA PAI)
WORKDIR /code

# 3. Copia e Instala as Dependências
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copia o Código da Aplicação
# Copia a pasta 'app' (que agora é um pacote)
COPY ./app /code/app

# 5. Expõe a Porta
EXPOSE 8000

# 6. Comando de Execução (Modo Módulo)
# Roda o MÓDULO app.main, e o objeto app
# Isso faz o Python tratar 'app' como um pacote.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
