# Aferição de Peformace para Cálculo de Matriz Inversa em Computador
# Programação por Thiago Barros

"""
Objetivo: O algoritmo calcula uma matriz aleatória para cada interação e retornará o uso da cpu a cada interação
e armazenará o dados da interação em um arquivo csv

"""

import numpy as np
import time
import psutil as ps
import pandas as pd

def calculo_matriz_inversa(linhas,colunas,interacoes):
    
    execucao = str(input("Deseja Executar o Cálculo? [Y,N]: "))
    
    if execucao in ["Y","y"]:
    
      tempo_total = 0
      interacao_atual = 0
      dados_saida = []
      
      
      for _ in range(interacoes):
          
        uso_cpu = ps.cpu_percent()
        frequencia_cpu = ps.cpu_freq().current
        matriz = np.random.rand(linhas, colunas)
        tempo_inicio = time.time()
        inversa_da_matriz = np.linalg.inv(matriz)
        tempo_final = time.time()
        tempo_passado = tempo_inicio- tempo_final
        tempo_total += tempo_passado
        interacao_atual += 1
        tempo_medio_interacao = tempo_total/ interacoes
        dados_entrada = [interacao_atual,round(tempo_medio_interacao,2),round(tempo_passado,2),round(tempo_final,2),uso_cpu,frequencia_cpu]
        dados_saida.append(dados_entrada)
        
        df = pd.DataFrame(dados_saida)
        df.to_csv("saida.csv")
    
    elif execucao in ["N","n"]:
        print("Cálculo Não Executado")
        pass
        
    

linhas = int(input("Entre com o valor de Linhas: "))
colunas = int(input("Entre com o valor de Colunas: "))
interacoes = int(input("Entre com o valor de Interações: "))
calculo_matriz_inversa(linhas, colunas,interacoes)
