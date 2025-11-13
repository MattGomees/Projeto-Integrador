# app/security_utils.py
import zipfile
import io  # Para manipulação de arquivos em memória
from typing import Tuple

import pandas as pd
from fastapi import UploadFile
from starlette.responses import StreamingResponse

# Define um nome padrão para os arquivos dentro do zip
DEFAULT_CSV_NAME = "predictions.csv"
DEFAULT_ZIP_NAME = "predictions.zip"

def unzip_csv_from_upload(upload_file: UploadFile) -> pd.DataFrame:

    # 1. Verifica se é um arquivo zip
    if upload_file.content_type not in ['application/zip', 'application/x-zip-compressed']:
        raise ValueError("Arquivo inválido. Por favor, envie um .zip.")

    # 2. Lê o conteúdo do UploadFile para um buffer em memória
    zip_buffer = io.BytesIO(upload_file.file.read())

    # 3. Abre o arquivo ZIP a partir do buffer
    with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
        
        # 4. Lista todos os arquivos dentro do ZIP
        all_files = zip_ref.namelist()
        
        # 5. Encontra o primeiro arquivo .csv
        csv_filename = None
        for f in all_files:
            if f.endswith('.csv'):
                csv_filename = f
                break
        
        if csv_filename is None:
            raise ValueError("Nenhum arquivo .csv encontrado dentro do .zip.")

        # 6. Lê o conteúdo do .csv para outro buffer em memória
        with zip_ref.open(csv_filename) as csv_file:
            # 7. Usa Pandas para ler o CSV e criar o DataFrame
            df = pd.read_csv(csv_file)
            return df

def zip_csv_to_response(df: pd.DataFrame) -> StreamingResponse:
    zip_buffer = io.BytesIO()
    csv_string = df.to_csv(index=False, encoding='utf-8')

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(DEFAULT_CSV_NAME, csv_string)

    zip_buffer.seek(0)

    # 5. Retorna a resposta
    return StreamingResponse(
        zip_buffer,
        media_type="application/x-zip-compressed",
        headers={
            'Content-Disposition': f'attachment; filename={DEFAULT_ZIP_NAME}'
        }
    )