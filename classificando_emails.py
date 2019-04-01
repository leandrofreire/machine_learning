#!-*- coding: utf8 -*-

import pandas as pd
classificacoes = pd.read_csv('emails.csv')
textosPuros = classificacoes['email']
textosQuebrados = textosPuros.str.lower().str.split(' ')
dicionario = set()


for lista in textosQuebrados:
    dicionario.update(lista)

totalDePalavras = len(dicionario)
tuplas = zip(dicionario, range(totalDePalavras))
tradutor = {palavra:indice for palavra, indice in tuplas}
print(tradutor['pode'])