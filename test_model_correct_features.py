#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para testar o modelo com os inputs corretos baseados nas features que ele espera.
"""

import os

import joblib
import pandas as pd


def test_model_with_correct_features():
    """Testa o modelo com as features corretas."""
    try:
        print("Tentando carregar o modelo...")
        
        model_path = "app/modelo_randomforest_match.pkl"
        if not os.path.exists(model_path):
            model_path = "/home/cazuza/workspace/datathon/decision/app/modelo_randomforest_match.pkl"
        
        model = joblib.load(model_path)
        print(f"Modelo carregado com sucesso de {model_path}")
        
        # Criar dados de teste com as features corretas
        test_data = pd.DataFrame({
            "modalidade": ["Remoto"],
            "formacao_e_idiomas.nivel_ingles": ["Avançado"],
            "comentario": ["Candidato com boas habilidades técnicas e experiência em projetos relevantes."],
            "cv_pt": ["Desenvolvedor com 5 anos de experiência em Python e frameworks web."]
        })
        
        print("\nDados de teste:")
        print(test_data)
        
        # Fazer previsão
        print("\nFazendo previsão...")
        
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(test_data)
            print(f"Probabilidade de sucesso: {prob[0][1]*100:.2f}%")
            
            # Avaliar o resultado
            if prob[0][1] > 0.7:
                print("✅ Alto potencial de sucesso!")
            elif prob[0][1] > 0.4:
                print("🔍 Potencial moderado. Verificar detalhes.")
            else:
                print("⚠️ Baixa probabilidade de sucesso para esta vaga.")
            
            print("\n✅ Teste concluído com sucesso!")
            return True
        else:
            pred = model.predict(test_data)
            print(f"Previsão: {'Sucesso' if pred[0] == 1 else 'Insucesso'}")
            print("\n✅ Teste concluído com sucesso!")
            return True
            
    except Exception as e:
        print(f"❌ Erro ao testar o modelo: {e}")
        return False

if __name__ == "__main__":
    test_model_with_correct_features()
