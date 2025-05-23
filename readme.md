# 🤖 IA para Otimização de Recrutamento - Decision

Este projeto foi desenvolvido durante o Datathon com o objetivo de aplicar Inteligência Artificial para otimizar o processo de recrutamento e seleção na empresa **Decision**, especializada em bodyshop e hunting de talentos na área de tecnologia.

## 🎯 Objetivo

Criar um MVP de modelo de machine learning capaz de prever o sucesso de um candidato, melhorando a eficiência no processo de seleção, com foco em:

- Redução de tempo no recrutamento
- Identificação de candidatos com alto potencial
- Padronização na análise de perfis

## 🛠️ Tecnologias utilizadas

- Python
- Scikit-learn
- Pandas / NumPy
- Category Encoders (TargetEncoder)
- Streamlit
- Joblib
- TF-IDF (caso campos de texto sejam usados)

## 📁 Estrutura do Projeto

```bash
📂 decision/
├── app/                             # Aplicação Streamlit
│   ├── app.py                       # Código principal da aplicação
│   └── modelo_randomforest_match.pkl # Modelo otimizado e comprimido
├── data/                            # Dados do projeto
│   ├── processed/                   # Dados processados
│   └── raw/                         # Dados brutos
├── models/                          # Modelos treinados
├── notebooks/                       # Notebooks de análise e treinamento
│   ├── notebook01.ipynb             # Notebook original
│   └── notebook01_reformulado.ipynb # Notebook reformulado
├── compress_model.py                # Script para comprimir o modelo
├── requirements.txt                 # Dependências do projeto
└── README.md                        # Este arquivo
```

## 🚀 Como usar este repositório

### Preparação do ambiente

1. Clone este repositório
```bash
git clone https://github.com/seu-usuario/decision.git
cd decision
```

2. Instale as dependências
```bash
pip install -r requirements.txt
```

3. Acessando o aplicativo
```bash
cd app
streamlit run app.py
```

### 📝 Notas sobre Git LFS

Este projeto utiliza Git LFS (Large File Storage) para gerenciar o modelo comprimido. Para trabalhar adequadamente com este repositório:

1. Instale o Git LFS
```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# macOS 
brew install git-lfs

# Windows (com Chocolatey)
choco install git-lfs
```

2. Configure o Git LFS antes de clonar ou após clonar
```bash
git lfs install
```

3. Para puxar todos os arquivos LFS após clonar
```bash
git lfs pull
```

### 🗜️ Compressão do Modelo

O modelo foi comprimido para facilitar o deploy, utilizando o script `compress_model.py`. O script reduz o tamanho do modelo em aproximadamente 70% usando compressão avançada:

```bash
python compress_model.py
```
