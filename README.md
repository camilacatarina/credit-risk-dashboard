# 🏦 Dashboard de Análise de Risco de Crédito

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

---

## Visão Geral

Este projeto é um **dashboard interativo** desenvolvido para auxiliar instituições financeiras na **avaliação de risco de crédito** de novos clientes. Utilizando um modelo de Machine Learning (Regressão Logística), a aplicação classifica o perfil de um cliente como **Baixo, Médio ou Alto risco**, com base em suas características financeiras e históricas.

A ferramenta foi construída com **Streamlit**, proporcionando uma interface amigável e intuitiva, e é ideal para tomadores de decisão que precisam de análises rápidas e baseadas em dados.

---

## Funcionalidades

- 📊 **Prévia dos Dados:** Visualização das primeiras linhas do dataset para entendimento inicial.
- 📈 **Métricas do Modelo:** Exibição da acurácia, quantidade de amostras e número de classes.
- 🔍 **Matriz de Confusão:** Gráfico que mostra o desempenho do modelo na classificação.
- 📉 **Análise Exploratória:** Gráficos interativos (histograma e boxplot) para explorar variáveis numéricas.
- 🔮 **Previsão para Novo Cliente:** Formulário dinâmico que permite inserir dados de um cliente e obter a classificação de risco em tempo real, com exibição das probabilidades.

---

## 🧠 Tecnologias Utilizadas

| Tecnologia | Finalidade |
|------------|------------|
| **Streamlit** | Criação do dashboard interativo e da interface web. |
| **Pandas & NumPy** | Manipulação e limpeza dos dados. |
| **Matplotlib & Seaborn** | Geração da matriz de confusão e gráficos estáticos. |
| **Plotly** | Criação de gráficos interativos (histograma, boxplot, barras de probabilidade). |
| **Scikit-learn** | Treinamento do modelo (LogisticRegression), pré-processamento (LabelEncoder, StandardScaler) e métricas (accuracy_score). |
| **LabelEncoder & StandardScaler** | Codificação de variáveis categóricas e padronização de variáveis numéricas. |

---

## 👤 Autora

Projeto desenvolvido por **Camila Catarina Pereira Chaves** – candidata ao programa de estágio Bradesco 2026.2.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/seu-perfil)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/SEU-USUARIO)

---

## ▶️ Como Executar o Projeto Localmente

Siga os passos abaixo para rodar o dashboard no seu computador.

### 📋 Pré-requisitos

- **Python 3.8 ou superior** – [Baixar Python](https://www.python.org/downloads/)
- **Git** (opcional, para clonar) – [Baixar Git](https://git-scm.com/downloads)
- Editor de código (VS Code, PyCharm, etc.)

---

### 📥 Clonar o Repositório

**Opção 1 – Clonar com Git:**

```bash
git clone https://github.com/SEU-USUARIO/credit-risk-dashboard.git
cd credit-risk-dashboard

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

pip install streamlit pandas numpy matplotlib seaborn scikit-learn scipy plotly

streamlit run app.py

http://localhost:8501 (Rodar na sua máquina)

## 🎯 Como Usar (Fluxo do Dashboard)

1. **Prévia dos Dados:** Ao abrir, você vê as primeiras linhas do dataset para ter uma noção dos dados.
2. **Métricas do Modelo:** Veja a acurácia do modelo treinado e a quantidade de dados.
3. **Matriz de Confusão:** Observe como o modelo está classificando corretamente os clientes.
4. **Análise Exploratória:** Selecione uma variável numérica no dropdown e visualize sua distribuição e outliers.
5. **Previsão para Novo Cliente:** Preencha os campos do formulário com os dados de um cliente fictício. Clique em **"Classificar Risco"** e veja a classificação e as probabilidades associadas a cada classe.

## Por que este projeto é relevante?

- ✅ **Aplicação prática de Machine Learning** em um problema real do setor financeiro.
- ✅ **Dashboard intuitivo** que pode ser usado por analistas de crédito sem conhecimento técnico.
- ✅ **Código limpo, comentado e com boas práticas** (separação de responsabilidades, tratamento de erros, validações).
- ✅ **Todo em português** para demonstrar cuidado com a experiência do usuário no contexto brasileiro.


**Projeto desenvolvido para porfólio e currículo**