#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para testar se o modelo RandomForest está funcionando corretamente.
"""

import os

import joblib
import pandas as pd


def test_model():
    """Tenta carregar o modelo e fazer uma previsão de teste."""
    try:
        print("Tentando carregar o modelo...")
        
        # Definir possíveis caminhos onde o modelo pode estar
        model_paths = [
            "app/modelo_randomforest_match.pkl",
            "models/modelo_randomforest_match.pkl",
            "/home/cazuza/workspace/datathon/decision/app/modelo_randomforest_match.pkl",
            "/home/cazuza/workspace/datathon/decision/models/modelo_randomforest_match.pkl"
        ]
        
        model = None
        for path in model_paths:
            if os.path.exists(path):
                print(f"Modelo encontrado em: {path}")
                model = joblib.load(path)
                break
        
        if model is None:
            print("Erro: Não foi possível encontrar o arquivo do modelo!")
            return

        # Criar dados de teste
        test_data = pd.DataFrame({
            "idade": [30],
            "experiencia": ["Pleno"],
            "formacao": ["Graduação"],
            "area_atuacao": ["Desenvolvimento"]
        })
        
        print("\nDados de teste:")
        print(test_data)
        
        # Fazer uma previsão
        print("\nTentando fazer uma previsão...")
        
        # Primeiro, verifique o tipo do modelo
        print(f"Tipo do modelo: {type(model)}")
        
        # Tente obter as features necessárias pelo modelo
        try:
            if hasattr(model, 'feature_names_in_'):
                print(f"Features esperadas pelo modelo: {model.feature_names_in_}")
            elif hasattr(model, 'feature_names_'):
                print(f"Features esperadas pelo modelo: {model.feature_names_}")
        except Exception as e:
            print(f"Não foi possível obter as features do modelo: {e}")
        
        # Tentar fazer previsão
        try:
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(test_data)
                print(f"Probabilidade de sucesso: {prob[0][1]*100:.2f}%")
            else:
                pred = model.predict(test_data)
                print(f"Previsão: {pred[0]}")
                
            print("\n✅ Teste concluído com sucesso!")
        except Exception as e:
            print(f"❌ Erro ao fazer previsão: {e}")
            
            # Imprimir mais detalhes para ajudar no diagnóstico
            print("\nDetalhes do modelo:")
            for attr in dir(model):
                if not attr.startswith('_'):  # Ignorar atributos privados
                    try:
                        value = getattr(model, attr)
                        if not callable(value):  # Não imprimir métodos
                            print(f"  - {attr}: {value}")
                    except Exception:
                        pass
    
    except Exception as e:
        print(f"❌ Erro ao testar o modelo: {e}")

if __name__ == "__main__":
    test_model()
