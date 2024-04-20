Attribute VB_Name = "Módulo3"
Option Explicit

'Teste de Referencias

Sub Teste()

    Worksheets("Planilha1").Range("A1").Value = 3.14159
    Worksheets("Planilha1").[A2] = 4
    [A5] = 4


End Sub
