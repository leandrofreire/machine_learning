import pandas as pd
from collections import Counter

df = pd.read_csv('situacao_do_cliente.csv')
X_df = df[['recencia', 'frequencia', 'semanas_de_inscricao']]
Y_df = df['situacao']

Xdummies_df = pd.get_dummies(X_df).astype(int)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_treino = 0.8

tamanho_treino = int(porcentagem_treino * len(Y))
#tamanho_validacao = len(Y) - tamanho_treino

treino_dados = X[:tamanho_treino]
treino_marcacoes = Y[:tamanho_treino]

validacao_dados = X[tamanho_treino:]
validacao_marcacoes = Y[tamanho_treino:]

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
modelo = OneVsRestClassifier(LinearSVC(random_state= 0))

k = 10
scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv= k)

import numpy as np
taxa_de_acerto = np.mean(scores)

print(scores)
print(taxa_de_acerto)