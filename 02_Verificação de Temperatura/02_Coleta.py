"""
Algoritmo de Coleta de Dados com Python
Programador: Thiago Barros

"""

"""
Observações:

Comandos para sensores de temperatura e velocidade da fan não foram implementadas para versão do windows,
sendo necessária outra biblioteca ou outro sistema operacional


"""

import serial
import time
from datetime import datetime
import pandas as pd
import psutil as ps

banco_de_dados = pd.DataFrame()

# Código Serial de Comunicação

porta_de_comunicacao= "COM6" 
baud_rate = 9600
ser = serial.Serial(porta_de_comunicacao, baud_rate)
time.sleep(2)


step = 0

try:
    if True:
        print("Coleta de Dados Iniciada")
        
        while True:
            step += 1
            temperatura_externa = ser.readline().decode('utf-8').rstrip()
           # temperatura_interna = ps.sensors_temperature()
            uso_cpu = ps.cpu_percent(interval=None)
            plano_energia = "Economia de Energia",
            dia_coleta = datetime.today().date()
            hora_coleta = datetime.now().strftime('%M:%S.%f')[:-4]
            #potencia_fan = ps.sensors_fans()
            frequencia_cpu = ps.cpu_freq(percpu=False).max
            
            saida = {"Step":step,
                     "Hora da Coleta":hora_coleta,
                     "Dia da Coleta":dia_coleta,
                    "Temperatura Externa":temperatura_externa,
                    #"Temperatura Interna":temperatura_interna,
                    #"Potência das Fans":potencia_fan,
                    "Frequencia da CPU":frequencia_cpu}
            
            database = pd.DataFrame([saida])  
            banco_de_dados = pd.concat([banco_de_dados, database], ignore_index=True)   
            
   
except KeyboardInterrupt:
    print("Coleta de Dados Encerrada")
    print(f"Foram coletadas {step} amostras")
    ser.close()
    banco_de_dados.to_csv('banco_de_dados.csv')
    
