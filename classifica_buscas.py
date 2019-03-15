import pandas as pd
from collections import Counter

df = pd.read_csv('busca.csv')
X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

Xdummies_df = pd.get_dummies(X_df).astype(int)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_treino = 0.9

tamanho_treino = int(porcentagem_treino * len(Y))
tamanho_teste = len(Y) - tamanho_treino

treino_dados = X[:tamanho_treino]
treino_marcacoes = Y[:tamanho_treino]

teste_dados = X[-tamanho_teste:]
teste_marcacoes = Y[-tamanho_teste:]

from sklearn.naive_bayes import MultinomialNB
modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)

acertos = resultado == teste_marcacoes

total_acertos = sum(acertos)
total_elementos = len(teste_dados)

taxa_acerto = 100.0 * total_acertos / total_elementos
print('Taxa de acerto do algoritmo %.2f' % taxa_acerto)
print('Total de elementos', total_elementos)


# Acerto base recebe o valor maximo que foi encontrado na soma do array
acerto_base = max(Counter(teste_marcacoes).values())
# Calcula a porcentagem de acerto, pega a quantidade encontrada no array e divide pelo tamanho
taxa_acerto_base = 100.0 * acerto_base / len(teste_marcacoes)
# retorna o maior entre eles utilizando função max()
print("Taxa de acerto base: %.2f" % taxa_acerto_base)