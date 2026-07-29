import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import os


st.set_page_config(
    page_title="Dashboard de Risco de Crédito",
    layout="wide",
    page_icon="🏦"
)

st.title("🏦 Dashboard de Análise de Risco de Crédito")
st.markdown("---")



def traduzir_colunas(df):
    """Traduz os nomes das colunas do inglês para o português"""
    mapa = {
        'ID': 'ID',
        'Customer_ID': 'ID_Cliente',
        'Month': 'Mês',
        'Name': 'Nome',
        'Age': 'Idade',
        'SSN': 'CPF',
        'Occupation': 'Profissão',
        'Annual_Income': 'Renda_Anual',
        'Monthly_Inhand_Salary': 'Salário_Mensal',
        'Num_Bank_Accounts': 'Qtde_Contas_Bancárias',
        'Num_Credit_Card': 'Qtde_Cartões_Crédito',
        'Interest_Rate': 'Taxa_Juros',
        'Num_of_Loan': 'Qtde_Empréstimos',
        'Type_of_Loan': 'Tipo_Empréstimo',
        'Delay_from_due_date': 'Atraso_Desde_Vencimento',
        'Num_of_Delayed_Payment': 'Qtde_Pagamentos_Atrasados',
        'Changed_Credit_Limit': 'Limite_Crédito_Alterado',
        'Num_Credit_Inquiries': 'Qtde_Consultas_Crédito',
        'Credit_Mix': 'Mistura_Crédito',
        'Outstanding_Debt': 'Dívida_Pendente',
        'Credit_Score': 'Pontuação_Crédito'
    }
    return df.rename(columns=mapa)

def traduzir_valores(df):

    meses = {
        'January': 'Janeiro', 'February': 'Fevereiro', 'March': 'Março',
        'April': 'Abril', 'May': 'Maio', 'June': 'Junho',
        'July': 'Julho', 'August': 'Agosto', 'September': 'Setembro',
        'October': 'Outubro', 'November': 'Novembro', 'December': 'Dezembro'
    }

    profissoes = {
        'Scientist': 'Cientista', 'Engineer': 'Engenheiro(a)',
        'Teacher': 'Professor(a)', 'Doctor': 'Médico(a)',
        'Lawyer': 'Advogado(a)', 'Manager': 'Gerente',
        'Analyst': 'Analista', 'Developer': 'Desenvolvedor(a)',
        'Designer': 'Designer', 'Student': 'Estudante'
    }

    emprestimos = {
        'Auto Loan': 'Empréstimo Automotivo',
        'Personal Loan': 'Empréstimo Pessoal',
        'Home Equity Loan': 'Empréstimo Imobiliário',
        'Credit-Builder Loan': 'Empréstimo para Construção de Crédito',
        'Student Loan': 'Empréstimo Estudantil',
        'Business Loan': 'Empréstimo Empresarial',
        'Mortgage': 'Financiamento Imobiliário'
    }

    credito = {
        'Good': 'Boa',
        'Standard': 'Média',
        'Poor': 'Ruim'
    }

    if 'Mês' in df.columns:
        df['Mês'] = df['Mês'].replace(meses)
    if 'Profissão' in df.columns:
        df['Profissão'] = df['Profissão'].replace(profissoes)
    if 'Tipo_Empréstimo' in df.columns:
        for eng, pt in emprestimos.items():
            df['Tipo_Empréstimo'] = df['Tipo_Empréstimo'].str.replace(eng, pt, regex=False)
    if 'Pontuação_Crédito' in df.columns:
        df['Pontuação_Crédito'] = df['Pontuação_Crédito'].replace(credito)

    return df

@st.cache_data
def carregar_dados():
    """Carrega o CSV e aplica limpeza e tradução"""
    df = pd.read_csv("train.csv", low_memory=False)


    if 'Monthly_Balance' in df.columns:
        df['Monthly_Balance'] = df['Monthly_Balance'].astype(str).str.replace(r'R\$', '', regex=False)
        df['Monthly_Balance'] = df['Monthly_Balance'].str.replace(',', '.').str.strip()
        df['Monthly_Balance'] = pd.to_numeric(df['Monthly_Balance'], errors='coerce')

 
    df = traduzir_colunas(df)

    df = traduzir_valores(df)

    return df


df = carregar_dados()


st.subheader("📋 Prévia dos Dados")
st.dataframe(df.head(10))
st.markdown("---")



@st.cache_resource
def treinar_modelo(df):
    """
    Pré-processa os dados, treina o modelo de Regressão Logística
    e retorna o modelo, scaler, métricas e o LabelEncoder usado.
    """
  
    df = df.dropna()

 
    alvo = 'Pontuação_Crédito'


    colunas_categoricas = df.select_dtypes(include=['object']).columns.tolist()
    if alvo in colunas_categoricas:
        colunas_categoricas.remove(alvo)


    le = LabelEncoder()
    for col in colunas_categoricas:
        df[col] = le.fit_transform(df[col].astype(str))


    df[alvo] = le.fit_transform(df[alvo].astype(str))


    X = df.drop(columns=[alvo])
    y = df[alvo]


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )


    modelo = LogisticRegression(max_iter=1000, random_state=42)
    modelo.fit(X_train, y_train)


    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)

    return modelo, scaler, X_test, y_test, y_pred, acuracia, le


with st.spinner("🔄 Treinando o modelo... Aguarde!"):
    modelo, scaler, X_teste, y_teste, y_predito, acuracia, le = treinar_modelo(df)


st.subheader("📊 Métricas do Modelo")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🎯 Acurácia", f"{acuracia:.2%}")
with col2:
    st.metric("📝 Amostras de Treino", f"{len(df)}")
with col3:
    st.metric("📊 Classes", f"{len(df['Pontuação_Crédito'].unique())}")
st.markdown("---")


st.subheader("🔍 Matriz de Confusão")
fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
cm = confusion_matrix(y_teste, y_predito)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
ax_cm.set_xlabel('Previsto')
ax_cm.set_ylabel('Real')
ax_cm.set_title('Matriz de Confusão')
st.pyplot(fig_cm)
st.markdown("---")


st.subheader("📈 Análise Exploratória")
colunas_numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if colunas_numericas:
    coluna_selecionada = st.selectbox("Selecione uma variável para visualizar:", colunas_numericas)
    col1, col2 = st.columns(2)
    with col1:
        fig_hist = px.histogram(df, x=coluna_selecionada, title=f"Distribuição de {coluna_selecionada}")
        st.plotly_chart(fig_hist, use_container_width=True)
    with col2:
        fig_box = px.box(df, y=coluna_selecionada, title=f"Boxplot de {coluna_selecionada}")
        st.plotly_chart(fig_box, use_container_width=True)
st.markdown("---")


st.subheader("🔮 Previsão de Risco para Novo Cliente")
st.markdown("Preencha os dados abaixo para classificar o risco de crédito do cliente:")

with st.form("formulario_predicao"):
    inputs = {}
    colunas_por_linha = 2
    colunas_input = st.columns(colunas_por_linha)


    colunas_features = [col for col in df.columns if col != 'Pontuação_Crédito']

    for i, col in enumerate(colunas_features):
        with colunas_input[i % colunas_por_linha]:
            if pd.api.types.is_numeric_dtype(df[col]):

                min_val = float(df[col].min())
                max_val = float(df[col].max())
                mean_val = float(df[col].mean())
                inputs[col] = st.number_input(
                    f"{col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    help=f"Valores entre {min_val:.2f} e {max_val:.2f}"
                )
            else:

                opcoes = df[col].dropna().unique().tolist()
                inputs[col] = st.selectbox(f"{col}", opcoes)

    submitted = st.form_submit_button("🔍 Classificar Risco")

if submitted:

    input_df = pd.DataFrame([inputs])


    for col in input_df.select_dtypes(include=['object']).columns:

        le_temp = LabelEncoder()

        valores_combinados = pd.concat([df[col], input_df[col]], axis=0).astype(str)
        le_temp.fit(valores_combinados)
        input_df[col] = le_temp.transform(input_df[col].astype(str))


    input_scaled = scaler.transform(input_df)


    predicao = modelo.predict(input_scaled)[0]
    probabilidades = modelo.predict_proba(input_scaled)[0]


    rotulos_risco = ['Baixo', 'Médio', 'Alto']

    st.success(f"### 🎯 Risco de Crédito: **{rotulos_risco[predicao]}**")


    prob_df = pd.DataFrame({
        'Classe': rotulos_risco,
        'Probabilidade': probabilidades
    })
    fig_prob = px.bar(prob_df, x='Classe', y='Probabilidade',
                      title="Probabilidades de Risco",
                      color='Classe',
                      color_discrete_sequence=['green', 'yellow', 'red'])
    st.plotly_chart(fig_prob, use_container_width=True)


st.markdown("---")
st.caption("💡 Projeto desenvolvido para currículo e portfólio - Camila Catarina Pereira Chaves")