import os
import pandas as pd
from dahuffman import HuffmanCodec
from collections import Counter
import json
import io

def _xor_cipher(data: bytes, key: str) -> bytes:
    key_bytes = key.encode('utf-8')
    key_len = len(key_bytes)
    return bytes(b ^ key_bytes[i % key_len] for i, b in enumerate(data))

def secure_file(file_path: str, output_name: str, key: str) -> (str, str):
    
    # 1. Ler os dados como string (Huffman na aula foi em string)
    with open(file_path, 'r', encoding='utf-8') as f:
        data_str = f.read()

    # 2. Calcular frequências (como em Compressão_de_Dados.ipynb)
    freq = Counter(data_str)
    
    # 3. Criar codec (como em Compressão_de_Dados.ipynb)
    codec = HuffmanCodec.from_frequencies(freq)
    
    # 4. Codificar dados
    encoded_data = codec.encode(data_str)
    
    # 5. Criptografar dados codificados
    encrypted_data = _xor_cipher(encoded_data, key)
    
    # 6. Salvar dados binários criptografados
    data_file_name = f"{output_name}.huff"
    with open(data_file_name, 'wb') as f:
        f.write(encrypted_data)
        
    # 7. Salvar tabela de frequência (necessário para descompressão)
    freq_file_name = f"{output_name}.freq.json"
    with open(freq_file_name, 'w', encoding='utf-8') as f:
        json.dump(dict(freq), f) 
        
    return data_file_name, freq_file_name

def unsecure_file(data_file_path: str, freq_file_path: str, key: str, output_csv_name: str) -> str:
    
    # 1. Ler a tabela de frequência
    with open(freq_file_path, 'r', encoding='utf-8') as f:
        freq_dict = json.load(f)
        freq = Counter(freq_dict)

    # 2. Recriar o codec (como em Compressão_de_Dados.ipynb)
    codec = HuffmanCodec.from_frequencies(freq)
    
    # 3. Ler os dados binários criptografados
    with open(data_file_path, 'rb') as f:
        encrypted_data = f.read()
        
    # 4. Descriptografar dados
    encoded_data = _xor_cipher(encrypted_data, key)
        
    # 5. Decodificar (como em Compressão_de_Dados.ipynb)
    decoded_str = codec.decode(encoded_data)
    
    # 6. Salvar string decodificada de volta para CSV
    with open(output_csv_name, 'w', encoding='utf-8') as f:
        f.write(decoded_str)
        
    return output_csv_name