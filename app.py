import streamlit as st
from utils import (
    carregar_dados, obter_numeros, frequencia_numeros,
    gerar_multiplas_sugestoes_estatisticas, gerar_jogo_neural,
    gerar_jogo_neural_multilabel, validar_modelo_neural_multilabel,
    calcular_acuracia_sugestao, salvar_sugestao, carregar_sugestoes, apagar_todas_sugestoes_salvas,
    exibir_sugestoes_salvas, adicionar_sorteio, exploracao_de_dados, estatisticas_soma,
)

senha_correta = st.secrets["auth"]["senha"]

if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    senha = st.text_input("🔒 Insira a chave de acesso:", type="password")
    if st.button("Entrar"):
        if senha == senha_correta:
            st.session_state.autenticado = True
            st.rerun()  # Atualização correta
        else:
            st.error("Chave incorreta.")
    st.stop()


configs_jogos = {
    "LotoFacil": {"min_num": 1, "max_num": 25, "num_bolas": 15},
    "MegaSena": {"min_num": 1, "max_num": 60, "num_bolas": 6},
    "Quina": {"min_num": 1, "max_num": 80, "num_bolas": 5},
    "Milionaria": {"min_num": 1, "max_num": 50, "num_bolas": 6},
    "DuplaSena": {"min_num": 1, "max_num": 50, "num_bolas": 6},
    "Timemania": {"min_num": 1, "max_num": 80, "num_bolas": 10},
    "DiaDeSorte": {"min_num": 1, "max_num": 31, "num_bolas": 7},
    "SuperSete": {"min_num": 0, "max_num": 9, "num_bolas": 7},
}

jogo_selecionado = st.sidebar.selectbox("Selecione o jogo", list(configs_jogos.keys()))

df = carregar_dados(jogo_selecionado)

if df is None:
    st.error("Erro ao carregar dados.")
    st.stop()

config = configs_jogos[jogo_selecionado]

aba = st.sidebar.selectbox("Selecione a aba", [
    "Exploração de Dados",
    "Sugestões Estatísticas",
    "Modelagem Neural Tradicional",
    "Modelagem Neural Multilabel",
    "Validação da Rede Neural",
    "Sugestões Salvas",
    "Adicionar Sorteio"
])

if df is None:
    st.error("Erro ao carregar dados.")
    st.stop()

bolas_df = obter_numeros(df)

if aba == "Exploração de Dados":
    st.title("Exploração de Dados")
    exploracao_de_dados(df)
    estatisticas_soma(df)

elif aba == "Sugestões Estatísticas":
    st.title("Sugestões Estatísticas")
    freq_series = frequencia_numeros(df)
    soma_jogos = bolas_df.sum(axis=1)
    media_soma = soma_jogos.mean()
    desvio_soma = soma_jogos.std()

    sugestoes_est = gerar_multiplas_sugestoes_estatisticas(
        freq_series, config["num_bolas"], media_soma, desvio_soma,
        config["min_num"], config["max_num"], n_sugestoes=5
    )
    for i, s in enumerate(sugestoes_est):
        acuracia = calcular_acuracia_sugestao(s, list(bolas_df.iloc[-1].values))
        st.write(f"Estatística {i+1}: {s} — Acertos: {acuracia*100:.2f}%")
        if st.button(f"Salvar Estatística {i+1}", key=f"estatistica_{i}"):
            salvar_sugestao(s, f"Estatística {i+1}", jogo_selecionado)

elif aba == "Modelagem Neural Tradicional":
    st.title("Modelagem Neural Tradicional (Regressor)")
    jogo_neural = gerar_jogo_neural(bolas_df, config)
    if jogo_neural:
        acuracia_neural = calcular_acuracia_sugestao(jogo_neural, list(bolas_df.iloc[-1].values))
        jogo_neural_int = list(map(int, jogo_neural))
        st.write(f"Sugestão Rede Neural: {jogo_neural_int} — Acertos: {acuracia_neural*100:.2f}%")
        if st.button("Salvar Rede Neural Tradicional"):
            salvar_sugestao(jogo_neural, "Rede Neural Tradicional", jogo_selecionado)

    st.subheader("Salvar jogo manualmente")
    numeros_manual = st.text_input(f"Digite {config['num_bolas']} números separados por vírgula:")
    if st.button("Salvar Jogo Manual"):
        try:
            numeros = [int(n.strip()) for n in numeros_manual.split(",")]
            if len(set(numeros)) != config["num_bolas"]:
                st.error(f"Insira exatamente {config['num_bolas']} números distintos.")
            elif any(n < config["min_num"] or n > config["max_num"] for n in numeros):
                st.error(f"Números devem estar entre {config['min_num']} e {config['max_num']}")
            else:
                salvar_sugestao(numeros, "Manual", jogo_selecionado)
                st.success("Sugestão manual salva com sucesso!")
        except Exception as e:
            st.error(f"Erro ao processar os números: {e}")



elif aba == "Modelagem Neural Multilabel":
    st.title("Modelagem Neural Multilabel")
    jogo_neural_multi = gerar_jogo_neural_multilabel(bolas_df, config)
    if jogo_neural_multi:
        acuracia_multi = calcular_acuracia_sugestao(jogo_neural_multi, list(bolas_df.iloc[-1].values))
        jogo_neural_multi_int = list(map(int, jogo_neural_multi))  # <- converte aqui
        st.write(f"Sugestão Rede Neural Multilabel: {jogo_neural_multi_int} — Acertos: {acuracia_multi*100:.2f}%")
        if st.button("Salvar Rede Neural Multilabel"):
            salvar_sugestao(jogo_neural_multi, "Rede Neural Multilabel", jogo_selecionado)

elif aba == "Validação da Rede Neural":
    st.title("Validação Temporal da Rede Neural Multilabel")
    n_validacoes = st.slider("Número de validações", 5, 30, 10)
    resultados = validar_modelo_neural_multilabel(bolas_df, config, n_validacoes=n_validacoes)

    # Cria um conjunto de tuplas com as combinações históricas já sorteadas
    historico_combinacoes = set(
        tuple(sorted(bolas_df.iloc[i].values)) for i in range(len(bolas_df))
    )

    resultados_formatados = []

    for i, (jogo_predito, acuracia) in enumerate(resultados):
        jogo_predito_int = tuple(sorted(map(int, jogo_predito)))  # converte e ordena para comparação
        ja_saiu = jogo_predito_int in historico_combinacoes
        resultados_formatados.append({
            "indice": n_validacoes - i,
            "jogo_predito": jogo_predito_int,
            "acuracia": acuracia,
            "ja_saiu": ja_saiu
        })

    # Ordena para mostrar primeiro os que ainda não saíram, depois os que já saíram,
    # e dentro desses grupos ordena pela maior acurácia
    resultados_formatados.sort(key=lambda x: (x["ja_saiu"], -x["acuracia"]))

    for res in resultados_formatados:
        label_saida = " (Já saiu)" if res["ja_saiu"] else " (Ainda não saiu)"
        acuracia_pct = res["acuracia"] * 100
        jogo_str = list(res['jogo_predito'])

        # Define cor de fundo
        if res["ja_saiu"]:
            bg_color = "#00ff3d"  # verde
        elif acuracia_pct < 50:
            bg_color = "#ff0017"  # vermelho
        else:
            bg_color = "white"

        texto = (
            f"<div style='color: {bg_color}; "
            "padding: 8px; border-radius: 5px; margin-bottom: 5px;'>"
            f"Validação -{res['indice']}: Predito: {jogo_str}, "
            f"Acertos: {acuracia_pct:.2f}%{label_saida}"
            "</div>"
        )
        st.markdown(texto, unsafe_allow_html=True)

    media = sum(r["acuracia"] for r in resultados_formatados) / len(resultados_formatados)
    st.markdown(f"**Acurácia média:** {media*100:.2f}%")

elif aba == "Sugestões Salvas":
    st.title("Sugestões Salvas")
    sugestoes = carregar_sugestoes()
    exibir_sugestoes_salvas(df, sugestoes, tipo_jogo_filtrar=jogo_selecionado)

    st.subheader("Gerenciar sugestões salvas")
    if st.button("Apagar todas as sugestões salvas"):
        apagar_todas_sugestoes_salvas()

elif aba == "Adicionar Sorteio":
    st.title("Adicionar Novo Sorteio")

    entrada_texto = st.text_input(
        f"Digite os números separados por vírgula (opcional):",
        value=""
    )

    numeros_iniciais = []
    if entrada_texto.strip():
        try:
            numeros_iniciais = [int(n.strip()) for n in entrada_texto.split(",")]
            # Valida quantidade e intervalo
            if len(numeros_iniciais) != config["num_bolas"]:
                st.error(f"Você deve informar exatamente {config['num_bolas']} números.")
                numeros_iniciais = []
            elif any(n < config["min_num"] or n > config["max_num"] for n in numeros_iniciais):
                st.error(f"Os números devem estar entre {config['min_num']} e {config['max_num']}.")
                numeros_iniciais = []
        except Exception:
            st.error("Erro na leitura dos números. Use apenas números inteiros separados por vírgula.")
            numeros_iniciais = []

    # Se não colou números válidos, preenche com valores vazios (None)
    if not numeros_iniciais:
        numeros_iniciais = [None] * config["num_bolas"]

    nums = []
    for i in range(config["num_bolas"]):
        n = st.number_input(
            f"Número {i+1}",
            min_value=config["min_num"],
            max_value=config["max_num"],
            value=numeros_iniciais[i] if numeros_iniciais[i] is not None else config["min_num"],
            key=f"num_input_{i}"
        )
        nums.append(n)

    if st.button("Adicionar Sorteio"):
        # Validar novamente antes de salvar
        if len(set(nums)) != config["num_bolas"]:
            st.error("Os números devem ser todos diferentes.")
        elif any(n < config["min_num"] or n > config["max_num"] for n in nums):
            st.error(f"Os números devem estar entre {config['min_num']} e {config['max_num']}.")
        else:
            df_novo = adicionar_sorteio(df, nums, f"pages/{jogo_selecionado}/base.xlsx", config)
            if df_novo is not None:
                st.success("Sorteio adicionado com sucesso! Recarregue a página.")
