# é godinho? tem perninha curta? faz auau?
# Lista (array) com valores para cada caracteristica
porco1 = [1, 1, 0]
porco2 = [1, 1, 0]
porco3 = [1, 1, 0]
cachorro4 = [1, 1, 1]
cachorro5 = [0, 1, 1]
cachorro6 = [0, 1, 1]

# Lista que recebe as listas com as caracteristicas
dados = [porco1, porco2, porco3, cachorro4, cachorro5, cachorro6]

# Marcações para testar e aprender
marcacoes = [1, 1, 1, -1, -1, -1]

#importar biblioteca de aprendizado
from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()
modelo.fit(dados, marcacoes)

# Essas listas é para verificar o aprendizado quando passado parametros para ver qual animal é
misterioso1 = [1, 1, 1]
misterioso2 = [1, 0, 0]
misterioso3 = [0, 0, 1]

# Recebe as listas dos dados misteriosos
teste = [misterioso1, misterioso2, misterioso3]

# Indica o que é cada elemento misterioso
marcacao_teste = [-1, 1, 1]

resultado = modelo.predict(teste)

diferencas = resultado - marcacao_teste

# Percorre o array diferenças e devolve os elementos que forem iguais a 0 e atribui e acertos
acertos = [d for d in diferencas if d == 0]

# Calcula o total de acertos e de elementos com a função len()
total_de_acertos = len(acertos)
total_de_elementos = len(teste)

# Calculo da taxa de acerto em %
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

print(resultado)
print(diferencas)
print(taxa_de_acerto)



