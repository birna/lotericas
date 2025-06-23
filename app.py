import streamlit as st
from streamlit_option_menu import option_menu
from utils import carregar_dados, exploracao_de_dados, modelagem_preditiva

# Configurações dos jogos
JOGOS_CONFIG = {
    "LotoFacil": {"min_num": 1, "max_num": 25, "num_bolas": 15},
    "MegaSena": {"min_num": 1, "max_num": 60, "num_bolas": 6},
    # Adicione outros jogos aqui
}

st.set_page_config(page_title="Análises de Loterias", layout="wide")
st.title("📊 Sistema de Análise de Loterias")

# Menu lateral
with st.sidebar:
    jogo_selecionado = option_menu(
        menu_title="Menu Principal",
        options=list(JOGOS_CONFIG.keys()),
        icons=["bar-chart-line"] * len(JOGOS_CONFIG),
        menu_icon="cast",
        default_index=0
    )

# Carregar dados do jogo selecionado
config_jogo = JOGOS_CONFIG[jogo_selecionado]
df = carregar_dados(jogo_selecionado)

if df is not None:
    # Abas internas
    aba = st.radio("Selecione a análise", ["Exploração de Dados", "Modelagem Preditiva"], horizontal=True)

    if aba == "Exploração de Dados":
        st.subheader(f"🔍 Exploração de Dados - {jogo_selecionado}")
        exploracao_de_dados(df)

    elif aba == "Modelagem Preditiva":
        st.subheader(f"🤖 Modelagem Preditiva - {jogo_selecionado}")
        modelagem_preditiva(df, config_jogo)
else:
    st.error("Erro ao carregar os dados do jogo.")
