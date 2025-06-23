import streamlit as st
from streamlit_option_menu import option_menu
from utils import (
    carregar_dados, obter_numeros, frequencia_numeros,
    gerar_multiplas_sugestoes_estatisticas, gerar_jogo_neural,
    salvar_sugestao, carregar_sugestoes, exibir_sugestoes_salvas,
    adicionar_sorteio, calcular_acuracia_sugestao
)

JOGOS_CONFIG = {
    "LotoFacil": {"min_num": 1, "max_num": 25, "num_bolas": 15},
    "MegaSena": {"min_num": 1, "max_num": 60, "num_bolas": 6},
}

st.set_page_config(page_title="Análises de Loterias", layout="wide")
st.title("📊 Sistema de Análise de Loterias")

with st.sidebar:
    jogo_selecionado = option_menu(
        "Menu Principal",
        options=list(JOGOS_CONFIG.keys()),
        icons=["bar-chart-line"] * len(JOGOS_CONFIG),
        menu_icon="cast",
        default_index=0
    )

df = carregar_dados(jogo_selecionado)
config = JOGOS_CONFIG[jogo_selecionado]
caminho_arquivo = f"pages/{jogo_selecionado}/data/base.xlsx"

if df is not None:
    aba = st.radio(
        "Selecione a análise",
        ["Exploração de Dados", "Modelagem Preditiva", "Sugestões Salvas", "Adicionar Sorteio"],
        horizontal=True
    )

    if aba == "Exploração de Dados":
        st.subheader(f"🔍 Exploração de Dados - {jogo_selecionado}")
        freq = frequencia_numeros(df)
        st.bar_chart(freq)
        ultimos = obter_numeros(df).tail(5)
        st.dataframe(ultimos.reset_index(drop=True))

    elif aba == "Modelagem Preditiva":
        st.subheader(f"🤖 Modelagem Preditiva - {jogo_selecionado}")
        bolas_df = obter_numeros(df)
        freq_series = frequencia_numeros(df)
        soma_jogos = bolas_df.sum(axis=1)
        media_soma = soma_jogos.mean()
        desvio_soma = soma_jogos.std()

        st.markdown("### Sugestões Estatísticas")
        sugestoes_est = gerar_multiplas_sugestoes_estatisticas(
            freq_series, config["num_bolas"], media_soma, desvio_soma,
            config["min_num"], config["max_num"], n_sugestoes=5
        )
        for i, s in enumerate(sugestoes_est):
            acuracia = calcular_acuracia_sugestao(s, list(bolas_df.iloc[-1].values))
            st.write(f"Estatística {i+1}: {s} — Acertos: {acuracia*100:.2f}%")
            if st.button(f"Salvar Estatística {i+1}", key=f"estatistica_{i}"):
                salvar_sugestao(s, f"estatística {i+1}", jogo_selecionado)

        st.markdown("### Sugestão Rede Neural")
        jogo_neural = gerar_jogo_neural(bolas_df, config)
        if jogo_neural:
            acuracia_neural = calcular_acuracia_sugestao(jogo_neural, list(bolas_df.iloc[-1].values))
            # converte np.int64 para int nativo
            jogo_neural_int = list(map(int, jogo_neural))
            st.write(f"{jogo_neural_int} — Acertos: {acuracia_neural*100:.2f}%")
            if st.button("Salvar Rede Neural"):
                salvar_sugestao(jogo_neural_int, "rede neural", jogo_selecionado)

    elif aba == "Sugestões Salvas":
        st.subheader("💾 Sugestões Salvas")
        sugestoes = carregar_sugestoes()
        if sugestoes:
            exibir_sugestoes_salvas(df, sugestoes, tipo_jogo_filtrar=jogo_selecionado)
        else:
            st.write("Nenhuma sugestão salva.")

    elif aba == "Adicionar Sorteio":
        st.subheader("➕ Adicionar Sorteio Futuro")

        num_bolas = config["num_bolas"]
        min_num = config["min_num"]
        max_num = config["max_num"]

        entrada = st.text_input(
            f"Digite os {num_bolas} números separados por vírgula (ex: 1, 5, 12, 23, ...)",
            placeholder="Ex: 3, 7, 12, 15, 18, 22, ..."
        )

        if st.button("Adicionar sorteio"):
            try:
                entrada_numeros = list(map(int, entrada.strip().split(",")))
                entrada_numeros = [n for n in entrada_numeros if min_num <= n <= max_num]

                if len(set(entrada_numeros)) != num_bolas:
                    st.error(f"⚠️ Insira exatamente {num_bolas} números **únicos** entre {min_num} e {max_num}.")
                else:
                    df = adicionar_sorteio(df, entrada_numeros, caminho_arquivo, config)
            except ValueError:
                st.error("⚠️ Formato inválido. Certifique-se de digitar apenas números separados por vírgulas.")


else:
    st.error("Erro ao carregar os dados do jogo.")
