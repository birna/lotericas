import random
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import MultiLabelBinarizer

import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def carregar_dados(jogo):
    caminho = f"pages/{jogo}/base.xlsx"
    try:
        df = pd.read_excel(caminho, na_values=["-", "", " "])
        for col in df.columns:
            if "data" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
                df = df.set_index(col)
                break
        for col in df.columns:
            temp = pd.to_numeric(df[col], errors="coerce")
            if temp.notna().mean() > 0.9:
                df[col] = temp.astype('Int64')
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados para {jogo}: {e}")
        return None

def obter_numeros(df):
    return df.filter(regex="(Bola|Coluna)", axis=1)

@st.cache_data
def frequencia_numeros(df):
    numeros = obter_numeros(df).values.flatten()
    numeros = pd.Series(numeros).dropna()
    numeros = numeros[numeros.apply(lambda x: str(x).isdigit())].astype(int)
    return numeros.value_counts().sort_index()

def verificar_se_jogo_ja_saiu(jogo_predito, bolas_df):
    jogo_predito_sorted = tuple(sorted(map(int, jogo_predito)))
    historico = set(tuple(sorted(map(int, row))) for row in bolas_df.values)
    return jogo_predito_sorted in historico

def gerar_jogo_completo(config, bolas_df=None):
    min_num = config.get("min_num", 1)
    max_num = config.get("max_num", 60)
    num_bolas = config.get("num_bolas", 6)
    jogo = sorted(random.sample(range(min_num, max_num + 1), num_bolas))
    resultado = {"numeros": jogo}

    if "quantidade_trevos_selecionar" in config:
        faixa = config["faixa_trevos_disponiveis"]
        qt = config["quantidade_trevos_selecionar"]
        resultado["trevos"] = sorted(random.sample(range(faixa[0], faixa[1]+1), qt))

    if "quantidade_meses_selecionar" in config:
        resultado["mes"] = random.randint(1, 12)

    if "quantidade_times_selecionar" in config:
        times = config.get("lista_times_disponiveis", [f"Time {i+1}" for i in range(80)])
        resultado["time"] = random.choice(times) if isinstance(times, list) else f"Time {random.randint(1, 80)}"

    if "quantidade_colunas" in config:
        qt = config["quantidade_colunas"]
        faixa = config["faixa_numeros_por_coluna"]
        resultado["colunas"] = [random.randint(faixa[0], faixa[1]) for _ in range(qt)]
        resultado["numeros"] = resultado["colunas"]

    if bolas_df is not None and "quantidade_colunas" not in config:
        resultado["ja_saiu"] = verificar_se_jogo_ja_saiu(resultado["numeros"], bolas_df)
    else:
        resultado["ja_saiu"] = False

    return resultado

def calcular_acuracia_sugestao(sugestao, ultimo_jogo):
    acertos = len(set(sugestao) & set(ultimo_jogo))
    total = len(ultimo_jogo)
    return acertos / total if total > 0 else 0

def gerar_jogo_neural_multilabel(bolas_df, config):
    min_num = config.get("min_num", 1)
    max_num = config.get("max_num", 60)
    num_bolas = config.get("num_bolas", 6)
    if len(bolas_df) < 20:
        st.warning("Dados insuficientes para treino do modelo.")
        return None
    X = bolas_df.iloc[:-1].values
    y_raw = bolas_df.iloc[1:].values
    mlb = MultiLabelBinarizer(classes=range(min_num, max_num+1))
    y = mlb.fit_transform(y_raw)
    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
    model.fit(X, y)
    probs = model.predict_proba(bolas_df.iloc[-1:].values)[0]
    indices_top = np.argsort(probs)[-num_bolas:]
    jogo = sorted([mlb.classes_[i] for i in indices_top])
    while len(jogo) < num_bolas:
        n = random.randint(min_num, max_num)
        if n not in jogo:
            jogo.append(n)
    return jogo

def validar_modelo_neural_multilabel(bolas_df, config, n_validacoes=10):
    min_num = config.get("min_num", 1)
    max_num = config.get("max_num", 60)
    num_bolas = config.get("num_bolas", 6)
    resultados = []

    mlb = MultiLabelBinarizer(classes=range(min_num, max_num + 1))

    for i in range(n_validacoes, 0, -1):
        train_end = -i - 1
        test_index = -i

        X_train = bolas_df.iloc[:train_end].values
        y_train_raw = bolas_df.iloc[1:train_end + 1].values
        y_train = mlb.fit_transform(y_train_raw)

        model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        X_test = bolas_df.iloc[test_index].values.reshape(1, -1)
        probs = model.predict_proba(X_test)[0]

        indices_top = np.argsort(probs)[-num_bolas:]
        numeros_preditos = sorted([mlb.classes_[idx] for idx in indices_top])

        sugestao = {"numeros": numeros_preditos}

        # Particularidades por jogo
        if "trevos" in config:
            qtd = config["trevos"]["qtd"]
            faixa_min, faixa_max = config["trevos"]["min"], config["trevos"]["max"]
            trevos = sorted(random.sample(range(faixa_min, faixa_max + 1), qtd))
            sugestao["trevos"] = trevos

        if "times" in config:
            times_disponiveis = config.get("lista_times_disponiveis", [f"Time {i+1}" for i in range(config["times"])])
            sugestao["time"] = random.choice(times_disponiveis)

        if "meses" in config:
            sugestao["mes"] = random.randint(1, config["meses"])

        if "colunas" in config:
            colunas = config["colunas"]
            faixa = config["faixa_coluna"]
            sugestao["colunas"] = [random.randint(faixa[0], faixa[1]) for _ in range(colunas)]
            sugestao["numeros"] = sugestao["colunas"]

        # Avaliação da acurácia
        jogo_real_idx = test_index + 1
        if jogo_real_idx >= len(bolas_df):
            jogo_real_idx = -1
        jogo_real = bolas_df.iloc[jogo_real_idx].values
        acuracia = calcular_acuracia_sugestao(numeros_preditos, list(jogo_real))

        resultados.append((sugestao, round(acuracia * 100, 2)))

    return resultados


def gerar_jogo_timemania(config):
    min_num = config["min_num"]
    max_num = config["max_num"]
    num_bolas = config["num_bolas"]
    lista_times = config.get("lista_times_disponiveis", [])

    numeros = []
    while len(numeros) < num_bolas:
        n = random.randint(min_num, max_num)
        if n not in numeros:
            numeros.append(n)
    numeros = sorted(numeros)

    time_do_coracao = random.choice(lista_times) if lista_times else "Time Padrão"
    return numeros, time_do_coracao

def gerar_jogo_milionaria(config):
    min_num = config["min_num"]
    max_num = config["max_num"]
    num_bolas = config["num_bolas"]

    numeros = []
    while len(numeros) < num_bolas:
        n = random.randint(min_num, max_num)
        if n not in numeros:
            numeros.append(n)
    numeros = sorted(numeros)

    trevos = []
    while len(trevos) < 2:
        t = random.randint(1, 6)
        if t not in trevos:
            trevos.append(t)
    trevos = sorted(trevos)

    return numeros, trevos

def gerar_jogo_supersete(config):
    num_colunas = config.get("num_bolas", 7)
    min_num = config.get("min_num", 0)
    max_num = config.get("max_num", 9)
    return [random.randint(min_num, max_num) for _ in range(num_colunas)]

def cor_fundo_sugestao(ja_saiu, acuracia_pct):
    if ja_saiu:
        return "#00ff3d"  # verde
    elif acuracia_pct < 50:
        return "#ff0017"  # vermelho
    else:
        return "white"

def gerar_jogo_estatistico(freq_series, num_bolas, media_soma, desvio_soma, min_num, max_num, seed=None):
    if seed is not None:
        random.seed(seed)
    candidatos = freq_series.head(num_bolas * 3).index.tolist()
    jogo = []
    tentativas = 0
    while len(jogo) < num_bolas and tentativas < 100:
        tentativas += 1
        num = random.choice(candidatos)
        if num not in jogo:
            jogo_temp = sorted(jogo + [num])
            soma_temp = sum(jogo_temp)
            pares_temp = sum(n % 2 == 0 for n in jogo_temp)
            impares_temp = len(jogo_temp) - pares_temp
            if (media_soma - desvio_soma <= soma_temp <= media_soma + desvio_soma) and (pares_temp >= 2 and impares_temp >= 2):
                jogo = jogo_temp
    while len(jogo) < num_bolas:
        num_aleatorio = random.randint(min_num, max_num)
        if num_aleatorio not in jogo:
            jogo.append(num_aleatorio)
            jogo = sorted(jogo)
    return jogo

def gerar_multiplas_sugestoes_estatisticas(freq_series, num_bolas, media_soma, desvio_soma, min_num, max_num, n_sugestoes=5):
    sugestoes = []
    for i in range(n_sugestoes):
        s = gerar_jogo_estatistico(freq_series, num_bolas, media_soma, desvio_soma, min_num, max_num, seed=i)
        sugestoes.append(s)
    return sugestoes

def gerar_jogo_neural(bolas_df, config):
    min_num = config.get("min_num", 1)
    max_num = config.get("max_num", 60)
    num_bolas = config.get("num_bolas", 6)

    if len(bolas_df) < 10:
        st.warning("Dados insuficientes para treino do modelo neural.")
        return None

    X = bolas_df.iloc[:-1].values
    y = bolas_df.sum(axis=1).iloc[1:].values

    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=0)
    model.fit(X[:-1], y[:-1])

    soma_prevista = model.predict(X[-1:].reshape(1, -1))[0]
    media = soma_prevista / num_bolas
    jogo_previsto = [int(round(media + i - (num_bolas // 2))) for i in range(num_bolas)]
    jogo_previsto = sorted(set(np.clip(jogo_previsto, min_num, max_num)))

    while len(jogo_previsto) < num_bolas:
        jogo_previsto.append(random.randint(min_num, max_num))
        jogo_previsto = sorted(set(jogo_previsto))

    return jogo_previsto

# Exploração de dados
def exploracao_de_dados(df, jogo, config):
    st.write("### Frequência dos números sorteados")
    freq_series = frequencia_numeros(df)
    st.bar_chart(freq_series)

    st.write("### Últimos 5 jogos")
    bolas = obter_numeros(df)
    st.dataframe(bolas.tail(5).reset_index(drop=True))

    st.write("### Estatísticas gerais dos números sorteados")
    numeros = obter_numeros(df).values.flatten()
    freq_series = pd.Series(numeros).value_counts().sort_index()
    st.write(f"- Total de números sorteados: {len(numeros)}")
    st.write(f"- Números únicos sorteados: {len(set(numeros))}")
    st.write(f"- Número mais frequente: {freq_series.idxmax()} ({freq_series.max()} vezes)")
    st.write(f"- Número menos frequente: {freq_series.idxmin()} ({freq_series.min()} vezes)")

    # Particularidades por jogo:
    if jogo == "Milionaria":
        st.write("### Frequência dos Trevos")
        freq_trevos = frequencia_trevos(df, config)
        if not freq_trevos.empty:
            st.bar_chart(freq_trevos)
        else:
            st.info("Sem dados válidos de trevos.")

    elif jogo == "Timemania":
        times_freq = frequencia_times_timemania(df)
        if not times_freq.empty:
            st.write("### Frequência dos Times do Coração")
            st.bar_chart(times_freq)
            st.write(f"Time mais sorteado: {times_freq.idxmax()} ({times_freq.max()} vezes)")
        else:
            st.write("Nenhuma informação de time disponível.")


def estatisticas_soma(df):
    bolas_df = obter_numeros(df)
    soma_jogos = bolas_df.sum(axis=1)
    media_soma = soma_jogos.mean()
    desvio_soma = soma_jogos.std()

    st.markdown("### Estatísticas da Soma dos Jogos")
    st.write(f"Média da soma: {media_soma:.2f}")
    st.write(f"Desvio padrão da soma: {desvio_soma:.2f}")
    fig, ax = plt.subplots()
    sns.histplot(soma_jogos, bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    # Último sorteio
    ultimo = list(map(int, bolas_df.iloc[-1].values))
    pares = sum(n % 2 == 0 for n in ultimo)
    impares = len(ultimo) - pares
    soma_ultimo = sum(ultimo)
    st.markdown("### Estatísticas do Último Sorteio")
    st.write(f"Números sorteados: {', '.join(map(str, ultimo))}")
    st.write(f"Soma: {soma_ultimo}, Par/Ímpar: {pares} / {impares}")

    return media_soma, desvio_soma

@st.cache_data
def frequencia_trevos(df, config):
    if "faixa_trevos_disponiveis" not in config:
        st.warning("Configuração de trevos não encontrada.")
        return pd.Series(dtype=int)

    faixa = config["faixa_trevos_disponiveis"]
    colunas_trevos = [col for col in df.columns if "Trevo" in col and df[col].dtype in [int, float, "Int64"]]

    if not colunas_trevos:
        st.info("Nenhuma coluna de Trevo encontrada.")
        return pd.Series(dtype=int)

    trevos = df[colunas_trevos].values.flatten()
    trevos = pd.to_numeric(pd.Series(trevos), errors="coerce").dropna().astype(int)

    trevos = trevos[(trevos >= faixa[0]) & (trevos <= faixa[1])]
    return trevos.value_counts().sort_index()


@st.cache_data
def frequencia_times_timemania(df):
    # Supondo que a coluna de time do coração se chama 'Time' ou similar
    col_time = [col for col in df.columns if "time" in col.lower()]
    if not col_time:
        return pd.Series(dtype=int)
    times = df[col_time[0]].dropna()
    return times.value_counts()
