
# 100 Dias de Python - Dia 1 - Pomodoro
# Computação Científica Para Engenharia
# Programa Desenvolvido por Thiago Barros
# Rev 00

import tkinter as tk
from tkinter import ttk
import time
import winsound

janela = tk.Tk()

def alerta():
    caminho_arquivo = "d.wav"
    winsound.PlaySound(caminho_arquivo, winsound.SND_ALIAS)


def iniciar_longo():
    global duracao_longo, inicio
    inicio = time.time()
    atualizar_longo()

def atualizar_longo():
    tempo_passado = time.time() - inicio
    tempo_restante = duracao_longo - tempo_passado

    if tempo_restante <= 0:
        cronometro.config(text="Tempo encerrado!")
        alerta()

    else:
        minutos, segundos = divmod(tempo_restante, 60)
        tempo_formatado = f"{int(minutos):02d}:{int(segundos):02d}"
        cronometro.config(text=tempo_formatado)
        cronometro.after(1000,atualizar_longo)

def iniciar_curto():
    global duracao_curto, inicio

    inicio = time.time()
    atualizar_curto()

def atualizar_curto():
    tempo_passado = time.time() - inicio
    tempo_restante = duracao_curto - tempo_passado

    if tempo_restante <= 0:
        cronometro2.config(text="Tempo encerrado!")
        alerta()
    else:
        minutos, segundos = divmod(tempo_restante, 60)
        tempo_formatado = f"{int(minutos):02d}:{int(segundos):02d}"
        cronometro2.config(text=tempo_formatado)
        cronometro2.after(1000, atualizar_curto)
        
def iniciar_rapido():
    global duracao_rapido, inicio

    inicio = time.time()
    atualizar_rapido()

def atualizar_rapido():
    tempo_passado = time.time() - inicio
    tempo_restante = duracao_rapido - tempo_passado

    if tempo_restante <= 0:
        cronometro3.config(text="Tempo \n encerrado!")
        alerta()
    else:
        minutos, segundos = divmod(tempo_restante, 60)
        tempo_formatado = f"{int(minutos):02d}:{int(segundos):02d}"
        cronometro3.config(text=tempo_formatado)
        cronometro3.after(1000,  atualizar_rapido)
        

duracao_longo = 2700
duracao_curto = 1500
duracao_rapido = 900


janela.geometry("580x200")
janela.maxsize(height=200, width=600)

janela.title("Pomodoro")
imagem = tk.PhotoImage(file="in.png")
janela.iconphoto(True, imagem)

estilo = ttk.Style()
estilo.configure("Estilo.TNotebook", tabmargins=[1, 1, 1, 1])
estilo.configure("Estilo.TNotebook.Tab", background="lightgray", padding=[10, 5])

notebook = ttk.Notebook(janela, style= "Estilo.TNotebook")

aba1 = ttk.Frame(notebook)
aba2 = ttk.Frame(notebook)
aba3 = ttk.Frame(notebook)
aba4 = ttk.Frame(notebook)


notebook.add(aba1, text="Longo - 45 Minutos")
notebook.add(aba2, text="Curto - 25 Minutos")
notebook.add(aba3, text="Rápido - 15 Minutos")
notebook.add(aba4, text="Informações")


fonte_texto = ("Helvetica, 55")
fonte_botao = ("Helvetica", 10)
fundo = "#eaeded"


button1 = tk.Button(aba1, text="Iniciar \n Contagem \n de Tempo", font = fonte_botao, width=12, height=9, bd = 4, bg =fundo , command=iniciar_longo)
button1.grid(row=0, column=0)

cronometro = tk.Label(aba1, font=fonte_texto, bd = 1, width=12,  )
cronometro.grid(row=0, column=1)

button2 = tk.Button(aba2, text="Iniciar \n Contagem \n de Tempo",  font = fonte_botao, width=12, height=9, bd = 4, bg = fundo , command=iniciar_curto)
button2.grid(row=0, column=0)

cronometro2 = tk.Label(aba2,  font=fonte_texto, bd = 1, width=12)
cronometro2.grid(row=0, column=1)

button3 = tk.Button(aba3, text="Iniciar \n Contagem \n de Tempo",  font = fonte_botao, width=12, height=9, bd = 4, bg = fundo , command=iniciar_rapido)
button3.grid(row=0, column=0)

cronometro3 = tk.Label(aba3,  font=fonte_texto, bd = 1, width=12)
cronometro3.grid(row=0, column=1)

text_info = "Pomodoro desenvolvido por Thiago Barros \n Python na Versão 3.10.11 "

informações = tk.Label(aba4, text=text_info, font=("Arial", 12) )
informações.pack(anchor="center", expand=True)


notebook.pack(expand=True, fill="both")

janela.mainloop()
