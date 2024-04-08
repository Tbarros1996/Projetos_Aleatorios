"""
PDF Fusion 1.0
Desenvolvido por Thiago Barros
Código Fonte Disponivel no Github: 
Desenvolvido em Python 3.11

"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import PyPDF2


class pdf_fusion(tk.Tk):
    
    def __init__(self):
        
        super().__init__()
        
        ico = tk.PhotoImage(file="pdf.png")
        
        self.lista_de_arquivos = []
        self.lista_de_nomes = []
        self.lista_de_locais = []
                
        self.title("PDF Fusion 1.0 - By Thiago Barros")
        self.geometry("600x300")
        self.resizable(False,False)
        self.iconphoto(True,ico)
        self.lista_pdf()
        self.botoes()
        self.barra()
        
    def lista_pdf(self):
        
        self.tabela = ttk.Treeview()
        self.tabela["columns"] = ("Nome do Arquivo", "Local")
        
        self.tabela.heading("#0",text="Ordem")
        self.tabela.heading("Nome do Arquivo",text="Nome do Arquivo")
        self.tabela.heading("Local",text="Local do Arquivo")
        self.tabela.pack(expand=True, fill="both")
        
        self.update()
      
      
    def update(self):
        
        self.tabela.delete(*self.tabela.get_children())
        
        for i, (lista_de_arquivos, lista_de_nomes, lista_de_locais) \
            in enumerate(zip(self.lista_de_arquivos, self.lista_de_nomes, self.lista_de_locais)):
            
                self.tabela.insert("", "end", text=str(i), values=(lista_de_nomes,lista_de_locais))
        
    def botoes(self):
        
        frame_botoes = tk.Frame(self)
        frame_botoes.pack()

        a = 3
        
        add_botao = tk.Button(frame_botoes, text="Adicionar PDF",command=self.add_arquivo, background="#00FA9A", height=a,borderwidth=3, relief="groove")
        add_botao.grid(row=0, column=0, padx=a)

        up_botao = tk.Button(frame_botoes, text="Para ↑",command = self.up, background="#DAA520",height=a, borderwidth=3, relief="groove")
        up_botao.grid(row=0, column=1, padx=a)

        down_botao = tk.Button(frame_botoes,text="Para ↓",command= self.down,background="#DAA520", height=a,borderwidth=3, relief="groove")
        down_botao.grid(row=0, column=2, padx=a)

        delete_botao = tk.Button(frame_botoes, text="Deletar PDF", command = self.deletar, background=	"#FA8072", height=a,borderwidth=3, relief="groove")
        delete_botao.grid(row=0, column=3, padx=a)

        merge_botao = tk.Button(frame_botoes, text="Unir PDFS", command = self.merger, background="#1E90FF",height=a,borderwidth=3, relief="groove")
        merge_botao.grid(row=0, column=4, pady=a)
        
        apagar_tudo = tk.Button(frame_botoes, text="Limpar Lista", command = self.clean,
        background = "#f0ff0a", height = a, borderwidth=3, relief = "groove")
        apagar_tudo.grid(row=0,column=5,pady=a)
        
    
    def barra(self):
        
        barra_rolagem = ttk.Scrollbar(self, orient = "vertical", command=self.tabela.yview)
        barra_rolagem.pack(side = "right",fill="y")
        self.tabela.configure(yscrollcommand=barra_rolagem.set)
                   
    def add_arquivo(self):
    
        arquivo_add = filedialog.askopenfilename(filetypes=[("Arquivos PDF", "*.pdf")])
        
        if arquivo_add:
            
            self.lista_de_arquivos.append(arquivo_add)
            self.lista_de_nomes.append(arquivo_add.split("/")[-1])
            self.lista_de_locais.append(arquivo_add)
                    
        self.update()       

    def up(self):
        
        indice_arquivo = self.tabela.selection()
        
        if indice_arquivo:
            
            indice_arquivo = int(self.tabela.index(indice_arquivo))
            
            if indice_arquivo > 0:

                self.lista_de_arquivos[indice_arquivo], self.lista_de_arquivos[indice_arquivo - 1] = \
                    self.lista_de_arquivos[indice_arquivo - 1], self.lista_de_arquivos[indice_arquivo]
                

                self.lista_de_nomes[indice_arquivo], self.lista_de_nomes[indice_arquivo - 1] = \
                    self.lista_de_nomes[indice_arquivo - 1], self.lista_de_nomes[indice_arquivo]
                
                self.lista_de_locais[indice_arquivo], self.lista_de_locais[indice_arquivo - 1] = \
                    self.lista_de_locais[indice_arquivo - 1], self.lista_de_locais[indice_arquivo]
                
                self.update()
                
    def down(self):
            
        indice_arquivo = self.tabela.selection()
        
        if indice_arquivo:
            
            indice_arquivo = int(self.tabela.index(indice_arquivo))
            
            if indice_arquivo < len(self.lista_de_arquivos) - 1 :
                

                self.lista_de_arquivos[indice_arquivo], self.lista_de_arquivos[indice_arquivo + 1] = \
                    self.lista_de_arquivos[indice_arquivo + 1], self.lista_de_arquivos[indice_arquivo]

                self.lista_de_nomes[indice_arquivo], self.lista_de_nomes[indice_arquivo + 1] = \
                    self.lista_de_nomes[indice_arquivo + 1], self.lista_de_nomes[indice_arquivo]
                
                self.lista_de_locais[indice_arquivo], self.lista_de_locais[indice_arquivo - 1] = \
                    self.lista_de_locais[indice_arquivo + 1], self.lista_de_locais[indice_arquivo]
                
                
                self.update()    
                            
            
    def deletar(self):
        
        indice_arquivo = self.tabela.selection()
        
        if indice_arquivo:
            
            indice_arquivo = int(self.tabela.index(indice_arquivo))
            
            del self.lista_de_arquivos[indice_arquivo]
            del self.lista_de_nomes[indice_arquivo]
            del self.lista_de_locais[indice_arquivo]
            
            self.update()
    
    
    def merger(self):
        
        merge_var = PyPDF2.PdfFileMerger()
        
        for file in self.lista_de_locais:
            merge_var.append(file)   
                       
        merge_var.write("new_pdf.pdf")
        merge_var.close()
        
            
    def clean(self):
        
        self.lista_de_arquivos =[]
        self.lista_de_nomes=[]
        self.lista_de_locais=[]
        
        self.update()
    
pdf_fusion = pdf_fusion()
pdf_fusion.mainloop()








