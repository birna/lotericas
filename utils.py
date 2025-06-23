import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.neural_network import MLPRegressor
import json

# --- Dados ---

def carregar_dados(jogo):
    caminho = f"pages/{jogo}/data/base.xlsx"
    try:
        df = pd.read_excel(caminho)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados para {jogo}: {e}")
        return None

def obter_numeros(df):
    return df.filter(regex="Bola", axis=1)

def frequencia_numeros(df):
    numeros = obter_numeros(df).values.flatten()
    return pd.Series(numeros).value_counts().sort_index()

# --- Análise e visualização ---

def exploracao_de_dados(df):
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

    # Estatísticas do último sorteio
    ultimo = list(map(int, bolas_df.iloc[-1].values))
    pares = sum(n % 2 == 0 for n in ultimo)
    impares = len(ultimo) - pares
    soma_ultimo = sum(ultimo)
    st.markdown("### Estatísticas do Último Sorteio")
    st.write(f"Números sorteados: {', '.join(map(str, ultimo))}")
    st.write(f"Soma: {soma_ultimo}, Par/Ímpar: {pares} / {impares}")

    return media_soma, desvio_soma

# --- Geração de jogos estatísticos ---

def gerar_jogo_estatistico(freq_series, num_bolas, media_soma, desvio_soma, min_num, max_num, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
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

# --- Modelagem neural ---

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

# --- Avaliação ---

def calcular_acuracia_sugestao(sugestao, ultimo_jogo):
    acertos = len(set(sugestao) & set(ultimo_jogo))
    total = len(ultimo_jogo)
    return acertos / total if total > 0 else 0

# --- Salvar / carregar sugestões ---

def salvar_sugestao(jogo, tipo_geracao, tipo_jogo, arquivo="sugestoes.txt"):
    sugestao = {
        "tipo": tipo_geracao,
        "jogo": jogo,
        "tipo_jogo": tipo_jogo,
    }
    with open(arquivo, "a", encoding="utf-8") as f:
        f.write(json.dumps(sugestao, ensure_ascii=False) + "\n")

def carregar_sugestoes():
    sugestoes = []
    try:
        with open("sugestoes.txt", "r", encoding="utf-8") as f:
            for linha in f:
                try:
                    sugestao = json.loads(linha.strip())
                    sugestao["jogo"] = list(map(int, sugestao["jogo"]))
                    if all(k in sugestao for k in ["tipo", "jogo", "tipo_jogo"]):
                        sugestoes.append(sugestao)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return sugestoes

def exibir_sugestoes_salvas(df, sugestoes, tipo_jogo_filtrar=None):
    bolas_df = obter_numeros(df)

    for sugestao in sugestoes:
        tipo = sugestao["tipo"]
        jogo = list(map(int, sugestao["jogo"]))
        tipo_jogo = sugestao["tipo_jogo"]

        if tipo_jogo_filtrar and tipo_jogo != tipo_jogo_filtrar:
            continue

        ultimo = list(map(int, bolas_df.iloc[-1].values))
        acertos = len(set(jogo) & set(ultimo))
        acuracia = acertos / len(ultimo)

        ja_saiu = False
        data_sorteio = None

        for idx, row in bolas_df.iterrows():
            sorteio = list(map(int, row.values))
            if sorted(sorteio) == sorted(jogo):
                ja_saiu = True
                if "Data Sorteio" in df.columns:
                    data_sorteio = df.loc[idx, "Data Sorteio"]
                break

        st.write(f"🔹 Tipo: {tipo}, Jogo: {jogo}, Tipo do jogo: {tipo_jogo}, Acertos: {acuracia*100:.2f}%")
        if ja_saiu:
            data_fmt = pd.to_datetime(data_sorteio).strftime('%d/%m/%Y') if data_sorteio else ""
            st.success(f"✅ Já foi sorteado em **{data_fmt}**.")
        else:
            st.info("🔍 Ainda **não foi sorteado**.")

# --- Adicionar sorteio ---

def adicionar_sorteio(df, numeros, caminho_arquivo, config):
    novo_registro = {col: pd.NA for col in df.columns}
    for i, num in enumerate(numeros):
        coluna_bola = f"Bola{i+1}"
        if coluna_bola in df.columns:
            novo_registro[coluna_bola] = num
        else:
            st.error(f"Coluna '{coluna_bola}' não encontrada no DataFrame.")
            return df
    coluna_data = "Data Sorteio"
    if coluna_data in df.columns:
        novo_registro[coluna_data] = pd.Timestamp.now().strftime("%d/%m/%Y")
    else:
        st.error(f"Coluna '{coluna_data}' não encontrada no DataFrame.")
        return df
    novo_df = pd.DataFrame([novo_registro])
    df_novo = pd.concat([df, novo_df], ignore_index=True)
    try:
        df_novo.to_excel(caminho_arquivo, index=False)
        st.success("Novo sorteio adicionado com sucesso!")
        return df_novo
    except Exception as e:
        st.error(f"Erro ao salvar arquivo Excel: {e}")
        return df

