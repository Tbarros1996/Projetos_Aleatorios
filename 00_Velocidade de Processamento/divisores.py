# 500 Exercícios Resolvidos com Python
# Thiago Barros
# Exercícios resolvidos com base no Livro - 500 Algoritimos Resolvidos (ANITA LOPES E GUTO GARCIA)
# Algoritimo Numero 215
# Capitulo 4

"""
Entrar com o número e imprimir todos os seus divisores

"""
# Nota: Você pode usar esse loop para calcular o tempo de resposta para cada interação

import time as tm
import pandas as pd
x = int(input("Entre com o Valor: "))
interacao = []
inicio = tm.time()
tempo =[]
inter = 0

for i in range(1, x+1):
    interacao.append(i-1)
    agora = tm.time()
    tempo.append(round(agora-inicio,2))
    if x % i == 0:
        print(i)

curva_interacao = pd.DataFrame({'Tempo de Processamento':tempo, 'Interacoes': interacao})
curva_interacao.to_csv("processamento.csv",index=False)