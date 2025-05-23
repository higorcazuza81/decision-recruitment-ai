import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Match IA - Recrutamento", layout="centered")
st.title("\U0001F916 IA para Match de Candidatos - Decision")

st.markdown("""
Este MVP usa um modelo de IA para prever a probabilidade de sucesso de um candidato com base em seu perfil.
""")

# Carregar modelo treinado
@st.cache_resource
def carregar_modelo():
    try:
        # Tenta carregar do caminho local para desenvolvimento
        return joblib.load("modelo_randomforest_match.pkl")
    except FileNotFoundError:
        # Caminho alternativo para produção em diferentes ambientes
        try:
            return joblib.load("app/modelo_randomforest_match.pkl")
        except FileNotFoundError:
            st.error("Erro ao carregar modelo. Verifique o arquivo .pkl")
            return None

modelo = carregar_modelo()

# Inputs adaptados conforme as variáveis esperadas pelo modelo
with st.form("formulario"):
    st.subheader("Preencha os dados do candidato")
    
    modalidade = st.selectbox("Modalidade de Trabalho", ["Presencial", "Híbrido", "Remoto"])
    nivel_ingles = st.selectbox("Nível de Inglês", ["Básico", "Intermediário", "Avançado", "Fluente", "Não informado"])
    comentario = st.text_area("Comentários sobre o candidato", "")
    cv_pt = st.text_area("Resumo do currículo", "")

    enviado = st.form_submit_button("Verificar Match")

if enviado:
    entrada = pd.DataFrame({
        "modalidade": [modalidade],
        "formacao_e_idiomas.nivel_ingles": [nivel_ingles],
        "comentario": [comentario],
        "cv_pt": [cv_pt]
    })

    prob = modelo.predict_proba(entrada)[0][1]
    st.markdown(f"### Probabilidade de sucesso: {prob*100:.2f}%")

    if prob > 0.7:
        st.success("\U0001F389 Alto potencial de sucesso!")
    elif prob > 0.4:
        st.info("\U0001F914 Potencial moderado. Verifique detalhes.")
    else:
        st.warning("\u26A0\ufe0f Baixa probabilidade de sucesso para esta vaga.")

st.markdown("""
---
Desenvolvido no Datathon - IA aplicada a Recrutamento | Decision
""")
