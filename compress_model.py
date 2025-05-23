#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para comprimir o modelo RandomForest para reduzir seu tamanho.
Usa o método de compressão do joblib para otimizar o tamanho do arquivo.
"""

import joblib
import os
import time
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def main():
    print("Iniciando compressão do modelo...")
    
    # Caminho do modelo original
    input_path = "models/modelo_randomforest_match.pkl"
    output_path = "app/modelo_randomforest_match_compressed.pkl"
    
    start = time.time()
    
    # Carregar o modelo
    print(f"Carregando modelo de {input_path}...")
    model = joblib.load(input_path)
    
    # Salvar com alta compressão
    print(f"Comprimindo e salvando em {output_path}...")
    joblib.dump(model, output_path, compress=9)
    
    # Verificando tamanhos
    original_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
    new_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    
    print(f"Compressão concluída em {time.time() - start:.2f} segundos")
    print(f"Tamanho original: {original_size:.2f} MB")
    print(f"Tamanho após compressão: {new_size:.2f} MB")
    print(f"Redução: {(1 - new_size/original_size)*100:.2f}%")
    
    # Substituir o modelo original no app
    os.replace(output_path, "app/modelo_randomforest_match.pkl")
    print("Modelo comprimido salvo em app/modelo_randomforest_match.pkl")

if __name__ == "__main__":
    main()
