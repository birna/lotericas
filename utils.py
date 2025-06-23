import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import json
import os

def apagar_todas_sugestoes_salvas():
    caminho = f"sugestoes.txt"
    if os.path.exists(caminho):
        os.remove(caminho)
        st.success(f"Arquivo de sugestões apagado: {caminho}")
    else:
        st.warning(f"Nenhum arquivo encontrado em: {caminho}")

@st.cache_data
def carregar_dados(jogo):
    caminho = f"pages/{jogo}/base.xlsx"
    st.info(f"Carregando dados de {caminho}...")

    try:
        df = pd.read_excel(caminho, na_values=["-", "", " "])

        # Coluna data para índice (primeiro, para manter a coluna e evitar alteração)
        for col in df.columns:
            if "data" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df = df.set_index(col)
                break

        # Detectar colunas numéricas (>=90% numérico) e converter sem dropar linhas
        for col in df.columns:
            temp = pd.to_numeric(df[col], errors="coerce")
            if temp.notna().mean() > 0.9:
                df[col] = temp.astype('Int64')  # Int64 permite valores NA

        st.write(f"Dados carregados com sucesso para o jogo {jogo}!")
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

# Exploração de dados
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

    # Último sorteio
    ultimo = list(map(int, bolas_df.iloc[-1].values))
    pares = sum(n % 2 == 0 for n in ultimo)
    impares = len(ultimo) - pares
    soma_ultimo = sum(ultimo)
    st.markdown("### Estatísticas do Último Sorteio")
    st.write(f"Números sorteados: {', '.join(map(str, ultimo))}")
    st.write(f"Soma: {soma_ultimo}, Par/Ímpar: {pares} / {impares}")

    return media_soma, desvio_soma

# Geração estatística
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

# Modelagem Rede Neural tradicional (MLPRegressor) - já existente
from sklearn.neural_network import MLPRegressor

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

# Nova modelagem Rede Neural Multilabel
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer

def gerar_jogo_neural_multilabel(bolas_df, config):
    min_num = config.get("min_num", 1)
    max_num = config.get("max_num", 60)
    num_bolas = config.get("num_bolas", 6)

    if len(bolas_df) < 20:
        st.warning("Dados insuficientes para treino do modelo neural multilabel (mínimo 20 registros).")
        return None

    X = bolas_df.iloc[:-1].values
    y_raw = bolas_df.iloc[1:].values

    mlb = MultiLabelBinarizer(classes=range(min_num, max_num + 1))
    y = mlb.fit_transform(y_raw)

    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42, alpha=0.001)
    model.fit(X, y)

    probs = model.predict_proba(bolas_df.iloc[-1:].values)[0]
    indices_top = np.argsort(probs)[-num_bolas:]
    jogo_previsto = sorted([mlb.classes_[i] for i in indices_top])

    while len(jogo_previsto) < num_bolas:
        n = random.randint(min_num, max_num)
        if n not in jogo_previsto:
            jogo_previsto.append(n)
            jogo_previsto.sort()

    return jogo_previsto

# Validação temporal para rede neural multilabel
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
        jogo_predito = sorted([mlb.classes_[idx] for idx in indices_top])

        jogo_real_idx = test_index + 1
        if jogo_real_idx >= len(bolas_df):
            jogo_real_idx = -1
        jogo_real = bolas_df.iloc[jogo_real_idx].values

        acuracia = calcular_acuracia_sugestao(jogo_predito, list(jogo_real))
        resultados.append((jogo_predito, acuracia))

    return resultados

def calcular_acuracia_sugestao(sugestao, ultimo_jogo):
    acertos = len(set(sugestao) & set(ultimo_jogo))
    total = len(ultimo_jogo)
    return acertos / total if total > 0 else 0

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
        acuracia = calcular_acuracia_sugestao(jogo, ultimo)

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

def adicionar_sorteio(df, numeros, caminho_arquivo, config):
    # Cria um novo registro com todas as colunas originais como NA
    novo_registro = {col: pd.NA for col in df.columns}

    # Preenche as colunas de Bola1, Bola2, ... com os números informados
    for i, num in enumerate(numeros):
        coluna_bola = f"Bola{i+1}"
        if coluna_bola in df.columns:
            novo_registro[coluna_bola] = num
        else:
            st.error(f"Coluna '{coluna_bola}' não encontrada no DataFrame.")
            return df

    # Preenche a coluna de data (se existir) com a data atual
    coluna_data = df.index.name  # Como data está no índice
    if coluna_data:
        novo_registro[coluna_data] = pd.Timestamp.now()

        # Como a data é índice, vamos adicionar a linha depois e setar índice correto
        novo_df = pd.DataFrame([novo_registro])
        novo_df.set_index(coluna_data, inplace=True)
        df_novo = pd.concat([df, novo_df])
    else:
        st.error("Coluna de data não encontrada para definir índice.")
        return df

    try:
        df_novo.to_excel(caminho_arquivo, index=True)  # Salva incluindo índice
        st.success("Novo sorteio adicionado com sucesso!")
        return df_novo
    except Exception as e:
        st.error(f"Erro ao salvar arquivo Excel: {e}")
        return df

