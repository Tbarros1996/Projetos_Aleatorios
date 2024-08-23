# Veneza Equipamentos
# Calculadora de Dados de Distância entre Coordenadas.
# Equação de Haversine
# Obs: Coordenadas em graus devem ser convertidas para radianos

from math import *
import numpy as np
import pandas as pd

caminho_arquivo = "~/Documentos/base.csv"

dados = pd.read_csv(caminho_arquivo)

latitude = np.array(dados["Latitude"])
longitude = np.array(dados["Longitude"])

shutdown = len(list(latitude)) - 1

raio_terra = 6.38 # em Km

i = 0
lista_delta =  []
rad = pi/180

while i < shutdown:

    inicial_lat = latitude[i] * rad
    final_lat = latitude[i + 1] * rad

    inicial_long = longitude[i] * rad
    final_long = longitude[i + 1] * rad

    a = (sin(final_lat - inicial_lat))**2
    b = cos(inicial_lat)*cos(final_lat)
    c = (sin(final_long - inicial_long))**2
    
    fator = a + b*c

    arco_tangente = atan2(sqrt(fator),sqrt(1 - fator))

    distancia = 2 * raio_terra * arco_tangente
    
    lista_delta.append(distancia)
    
    i += 1

print(f"Distância Percorrida: {sum(lista_delta)} Km")

    




















