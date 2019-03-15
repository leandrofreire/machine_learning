import pandas as pd
from collections import Counter

df = pd.read_csv('busca.csv')
# Teste 1: home, busca, logado
# Teste 2: home, busca
# Teste 3: home, logado
# Teste 4: busca, logado
# Teste 5: busca (85,7% 7 testes)

X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

Xdummies_df = pd.get_dummies(X_df).astype(int)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_treino = 0.8
porcentagem_teste = 0.1

tamanho_treino = int(porcentagem_treino * len(Y))
tamanho_teste = int(porcentagem_teste * len(Y))
tamanho_validacao = len(Y) - tamanho_treino - tamanho_teste

# 0 até 799
treino_dados = X[:tamanho_treino]
treino_marcacoes = Y[:tamanho_treino]

# 800 até 899
fim_de_teste = tamanho_treino + tamanho_teste
teste_dados = X[tamanho_treino: fim_de_teste]
teste_marcacoes = Y[tamanho_treino: fim_de_teste]

# 900 até 999
validacao_dados = X[fim_de_teste:]
validacao_marcacoes = Y[fim_de_teste:]

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
    modelo.fit(treino_dados, treino_marcacoes)

    resultado = modelo.predict(teste_dados)

    acertos = resultado == teste_marcacoes

    total_acertos = sum(acertos)
    total_elementos = len(teste_dados)

    taxa_acerto = 100.0 * total_acertos / total_elementos

    msg = "Taxa de acerto do algoritmo {0}: {1}".format(nome, taxa_acerto)
    print(msg)
    return taxa_acerto


def teste_real(modelo, validacao_dados, validacao_marcacoes):
    resultado = modelo.predict(validacao_dados)
    acertos = resultado == validacao_marcacoes

    total_acertos = sum(acertos)
    total_elementos = len(validacao_marcacoes)
    taxa_acerto = 100.0 * total_acertos / total_elementos
    msg = "Taxa de acerto do vencedor entre os dois algoritmos no mundo real: {0}".format(taxa_acerto)
    print(msg)

from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

if resultadoMultinomial > resultadoAdaBoost:
    vencedor = modeloMultinomial
else:
    vencedor = modeloAdaBoost

teste_real(vencedor, validacao_dados, validacao_marcacoes)


# Acerto base recebe o valor maximo que foi encontrado na soma do array
acerto_base = max(Counter(validacao_marcacoes).values())
# Calcula a porcentagem de acerto, pega a quantidade encontrada no array e divide pelo tamanho
taxa_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
# retorna o maior entre eles utilizando função max()
print("Taxa de acerto base: %.2f" % taxa_acerto_base)

total_elementos = len(teste_dados)
print('Total de elementos', total_elementos)