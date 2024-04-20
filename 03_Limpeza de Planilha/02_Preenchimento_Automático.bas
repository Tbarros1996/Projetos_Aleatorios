Attribute VB_Name = "Módulo1"
'Algoritmo Para Preencher Linhas Vazias Com Base em uma Coluna Preenchida

Sub preencher()

    'x = Célula de Referencia Para a Coluna de Preenchimento
    
    x = "G1"
    
    On Error Resume Next
    
    Range(x).CurrentRegion.SpecialCells(xlCellTypeBlanks).FormulaR1C1 = "=R[-1]C"
    Range(x).CurrentRegion.Value = Range(x).CurrentRegion.Value
    
End Sub
