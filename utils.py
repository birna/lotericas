import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from collections import Counter
from sklearn.neural_network import MLPRegressor

def carregar_dados(jogo):
    caminho = f"pages/{jogo}/data/base.xlsx"
    try:
        df = pd.read_excel(caminho)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

def obter_numeros(df):
    return df.filter(regex="Bola", axis=1)

def frequencia_numeros(df):
    numeros = obter_numeros(df).values.flatten()
    return pd.Series(numeros).value_counts().sort_index()

def ultimos_jogos(df, n=5):
    bolas = obter_numeros(df)
    st.dataframe(bolas.tail(n).reset_index(drop=True))

def exploracao_de_dados(df):
    st.write("### Frequência dos números sorteados")
    freq = obter_numeros(df).values.flatten()
    freq_series = pd.Series(freq).value_counts().sort_index()
    st.bar_chart(freq_series)

    st.write("### Últimos 5 jogos")
    ultimos_jogos(df)

    st.write("### Estatísticas gerais dos números sorteados")
    numeros = obter_numeros(df).values.flatten()
    st.write(f"- Total de números sorteados: {len(numeros)}")
    st.write(f"- Números únicos sorteados: {len(set(numeros))}")
    st.write(f"- Número mais frequente: {freq_series.idxmax()} ({freq_series.max()} vezes)")
    st.write(f"- Número menos frequente: {freq_series.idxmin()} ({freq_series.min()} vezes)")

def modelagem_preditiva(df, config):
    bolas_df = obter_numeros(df)
    min_num = config.get("min_num", 1)
    max_num = config.get("max_num", 60)
    num_bolas = config.get("num_bolas", 6)

    freq = bolas_df.values.flatten()
    freq_series = pd.Series(freq).value_counts().sort_values(ascending=False)

    # Frequência dos números - base para geração de jogos
    st.markdown("### Frequência dos números (base para geração de jogos)")
    st.bar_chart(freq_series.head(20))

    # Estatísticas da soma dos jogos
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

    # Geração de jogos com restrições estatísticas
    st.markdown("### Geração de jogos com restrições estatísticas")
    def gerar_jogo():
        candidatos = freq_series.head(num_bolas * 3).index.tolist()
        jogo = []
        tentativas = 0
        while len(jogo) < num_bolas and tentativas < 100:
            tentativas += 1
            num = np.random.choice(candidatos)
            if num not in jogo:
                jogo_temp = sorted(jogo + [num])
                soma_temp = sum(jogo_temp)
                pares_temp = sum(n % 2 == 0 for n in jogo_temp)
                impares_temp = len(jogo_temp) - pares_temp
                # Filtra soma e paridade para equilíbrio
                if (media_soma - desvio_soma <= soma_temp <= media_soma + desvio_soma) and (pares_temp >= 2 and impares_temp >= 2):
                    jogo = jogo_temp
        # Completa se não conseguir cumprir restrições
        while len(jogo) < num_bolas:
            num_aleatorio = np.random.randint(min_num, max_num + 1)
            if num_aleatorio not in jogo:
                jogo.append(num_aleatorio)
                jogo = sorted(jogo)
        return jogo

    jogo_gerado = gerar_jogo()
    st.write(f"🎲 Jogo sugerido: {', '.join(map(str, jogo_gerado))}")

    # Previsão simples da soma para o próximo sorteio usando MLPRegressor
    st.markdown("### Previsão da soma para próximo sorteio (MLPRegressor)")
    X = bolas_df.iloc[:-1].values
    y = soma_jogos.iloc[1:].values
    if len(X) > 10:
        model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=0)
        model.fit(X[:-1], y[:-1])
        soma_prevista = model.predict(X[-1:].reshape(1, -1))[0]
        st.write(f"🎯 Soma prevista para o próximo sorteio: {soma_prevista:.2f}")
    else:
        st.write("Dados insuficientes para previsão confiável.")
