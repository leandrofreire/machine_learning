from dados import carregar_acessos

X, Y = carregar_acessos()

# Variaveispara testar o modelo
treino_dados = X[:90]
treino_marcacoes = Y[:90]

# 10% de Variaveis para testar o modelo (são as últimas linhas do array)
teste_dados = X[-9:]
teste_marcacoes = Y[-9:]

from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()
# Fit vai treinar os elementos
modelo.fit(treino_dados, treino_marcacoes)
# resultado recebe a predição dos elementos em X de teste
resultado = modelo.predict(teste_dados)
# A diferença é a subtração entre o resultado da predição com os valores de Y
diferencas = resultado - teste_marcacoes
# Pega a quantidade de acertos e atribui a acertos
acertos = [d for d in diferencas if d == 0]
total_de_acertos = len(acertos)
total_de_elementos = len(teste_dados)

taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

print(round(taxa_de_acerto))
print(total_de_elementos)