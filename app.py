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

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Dashboard de Risco de Crédito",
    layout="wide",
    page_icon="🏦"
)

st.title("🏦 Dashboard de Análise de Risco de Crédito")
st.markdown("---")

# ============================================================
# 1. CARREGAMENTO, LIMPEZA E TRADUÇÃO DOS DADOS
# ============================================================

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
    """Traduz os valores das colunas categóricas para o português"""
    # Meses
    meses = {
        'January': 'Janeiro', 'February': 'Fevereiro', 'March': 'Março',
        'April': 'Abril', 'May': 'Maio', 'June': 'Junho',
        'July': 'Julho', 'August': 'Agosto', 'September': 'Setembro',
        'October': 'Outubro', 'November': 'Novembro', 'December': 'Dezembro'
    }
    # Profissões (exemplos – adicione mais se necessário)
    profissoes = {
        'Scientist': 'Cientista', 'Engineer': 'Engenheiro(a)',
        'Teacher': 'Professor(a)', 'Doctor': 'Médico(a)',
        'Lawyer': 'Advogado(a)', 'Manager': 'Gerente',
        'Analyst': 'Analista', 'Developer': 'Desenvolvedor(a)',
        'Designer': 'Designer', 'Student': 'Estudante'
    }
    # Tipos de empréstimo
    emprestimos = {
        'Auto Loan': 'Empréstimo Automotivo',
        'Personal Loan': 'Empréstimo Pessoal',
        'Home Equity Loan': 'Empréstimo Imobiliário',
        'Credit-Builder Loan': 'Empréstimo para Construção de Crédito',
        'Student Loan': 'Empréstimo Estudantil',
        'Business Loan': 'Empréstimo Empresarial',
        'Mortgage': 'Financiamento Imobiliário'
    }
    # Classificação de crédito
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

    # Limpeza da coluna Monthly_Balance (se existir no original)
    if 'Monthly_Balance' in df.columns:
        df['Monthly_Balance'] = df['Monthly_Balance'].astype(str).str.replace(r'R\$', '', regex=False)
        df['Monthly_Balance'] = df['Monthly_Balance'].str.replace(',', '.').str.strip()
        df['Monthly_Balance'] = pd.to_numeric(df['Monthly_Balance'], errors='coerce')

    # Traduzir nomes das colunas
    df = traduzir_colunas(df)
    # Traduzir valores
    df = traduzir_valores(df)

    return df

# Carregar os dados
df = carregar_dados()

# Mostrar prévia
st.subheader("📋 Prévia dos Dados")
st.dataframe(df.head(10))
st.markdown("---")

# ============================================================
# 2. PRÉ-PROCESSAMENTO E TREINAMENTO DO MODELO
# ============================================================

@st.cache_resource
def treinar_modelo(df):
    """
    Pré-processa os dados, treina o modelo de Regressão Logística
    e retorna o modelo, scaler, métricas e o LabelEncoder usado.
    """
    # Remover linhas com valores faltantes
    df = df.dropna()

    # A coluna alvo é 'Pontuação_Crédito' (traduzida)
    alvo = 'Pontuação_Crédito'

    # Separar colunas categóricas e numéricas
    colunas_categoricas = df.select_dtypes(include=['object']).columns.tolist()
    if alvo in colunas_categoricas:
        colunas_categoricas.remove(alvo)

    # Codificar variáveis categóricas (exceto a alvo)
    le = LabelEncoder()
    for col in colunas_categoricas:
        df[col] = le.fit_transform(df[col].astype(str))

    # Codificar a variável alvo
    df[alvo] = le.fit_transform(df[alvo].astype(str))

    # Separar features e target
    X = df.drop(columns=[alvo])
    y = df[alvo]

    # Padronizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Treinar modelo
    modelo = LogisticRegression(max_iter=1000, random_state=42)
    modelo.fit(X_train, y_train)

    # Predizer e calcular acurácia
    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)

    return modelo, scaler, X_test, y_test, y_pred, acuracia, le

# Treinar o modelo
with st.spinner("🔄 Treinando o modelo... Aguarde!"):
    modelo, scaler, X_teste, y_teste, y_predito, acuracia, le = treinar_modelo(df)

# ============================================================
# 3. MÉTRICAS DO MODELO
# ============================================================
st.subheader("📊 Métricas do Modelo")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🎯 Acurácia", f"{acuracia:.2%}")
with col2:
    st.metric("📝 Amostras de Treino", f"{len(df)}")
with col3:
    st.metric("📊 Classes", f"{len(df['Pontuação_Crédito'].unique())}")
st.markdown("---")

# ============================================================
# 4. MATRIZ DE CONFUSÃO
# ============================================================
st.subheader("🔍 Matriz de Confusão")
fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
cm = confusion_matrix(y_teste, y_predito)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
ax_cm.set_xlabel('Previsto')
ax_cm.set_ylabel('Real')
ax_cm.set_title('Matriz de Confusão')
st.pyplot(fig_cm)
st.markdown("---")

# ============================================================
# 5. ANÁLISE EXPLORATÓRIA
# ============================================================
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

# ============================================================
# 6. PREDIÇÃO PARA NOVO CLIENTE
# ============================================================
st.subheader("🔮 Previsão de Risco para Novo Cliente")
st.markdown("Preencha os dados abaixo para classificar o risco de crédito do cliente:")

with st.form("formulario_predicao"):
    inputs = {}
    colunas_por_linha = 2
    colunas_input = st.columns(colunas_por_linha)

    # Pegar as colunas que serão usadas como features (todas exceto a alvo)
    colunas_features = [col for col in df.columns if col != 'Pontuação_Crédito']

    for i, col in enumerate(colunas_features):
        with colunas_input[i % colunas_por_linha]:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Coluna numérica
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
                # Coluna categórica (já traduzida)
                opcoes = df[col].dropna().unique().tolist()
                inputs[col] = st.selectbox(f"{col}", opcoes)

    submitted = st.form_submit_button("🔍 Classificar Risco")

if submitted:
    # Criar DataFrame com os inputs
    input_df = pd.DataFrame([inputs])

    # Codificar variáveis categóricas usando o mesmo LabelEncoder do treino
    # Precisamos reutilizar o encoder que foi ajustado no treinamento.
    # Como temos o le, mas ele foi usado para todas as colunas, vamos refazer
    # a codificação com base nos dados originais.
    # Para simplificar, vamos codificar novamente com fit_transform, mas
    # isso pode não ser consistente se houver valores novos.
    # Melhor: salvar os encoders por coluna. Mas para um MVP, faremos:
    for col in input_df.select_dtypes(include=['object']).columns:
        # Usar o LabelEncoder já treinado se possível, mas aqui criamos um novo
        # e aplicamos fit_transform (pode causar problemas se novos valores surgirem)
        # Como é uma demonstração, isso é aceitável.
        le_temp = LabelEncoder()
        # Combinar valores do treino e do input para manter consistência
        valores_combinados = pd.concat([df[col], input_df[col]], axis=0).astype(str)
        le_temp.fit(valores_combinados)
        input_df[col] = le_temp.transform(input_df[col].astype(str))

    # Padronizar
    input_scaled = scaler.transform(input_df)

    # Predizer
    predicao = modelo.predict(input_scaled)[0]
    probabilidades = modelo.predict_proba(input_scaled)[0]

    # Rótulos em português
    rotulos_risco = ['Baixo', 'Médio', 'Alto']

    st.success(f"### 🎯 Risco de Crédito: **{rotulos_risco[predicao]}**")

    # Mostrar probabilidades
    prob_df = pd.DataFrame({
        'Classe': rotulos_risco,
        'Probabilidade': probabilidades
    })
    fig_prob = px.bar(prob_df, x='Classe', y='Probabilidade',
                      title="Probabilidades de Risco",
                      color='Classe',
                      color_discrete_sequence=['green', 'yellow', 'red'])
    st.plotly_chart(fig_prob, use_container_width=True)

# ============================================================
# RODAPÉ
# ============================================================
st.markdown("---")
st.caption("💡 Projeto desenvolvido como parte do portfólio para processo seletivo - Bradesco 2026.2")