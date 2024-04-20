Attribute VB_Name = "Módulo1"
' Teste - Verificando Aprovados


Sub Aprovados()

Ultima_Celula = Cells(Rows.Count, 9).End(xlUp).Row

For i = 1 To Ultima_Celula

    If Cells(i, 9) >= 40 Then

    Cells(i, 10) = "APROVADO"
    
    Else
    
    Cells(i, 10) = "REPROVADO"
    
    End If
    
Next i



End Sub
