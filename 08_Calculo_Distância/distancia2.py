# Veneza Equipamentos
# Calculadora de Dados de Distância entre Coordenadas.
# Padrão UTM

from math import *
import numpy as np
import pandas as pd

caminho_arquivo = "~/Documentos/base.csv"

dados = pd.read_csv(caminho_arquivo)

latitude = np.array(dados["Latitude"])
longitude = np.array(dados["Longitude"])

shutdown = len(list(latitude)) - 1

i = 0
lista_delta =  []

while i < shutdown:

    inicial_lat = latitude[i] 
    final_lat = latitude[i + 1] 

    inicial_long = longitude[i] 
    final_long = longitude[i + 1]

    euclidiana = sqrt((final_lat-inicial_lat)**2 + (final_long-inicial_long)**2)

    lista_delta.append(euclidiana)

    i += 1
    
print(sum(lista_delta))

    




















