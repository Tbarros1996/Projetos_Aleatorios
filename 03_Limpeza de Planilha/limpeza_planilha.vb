' Algoritmo de Limpeza de Dados
' Rev 01 - 22/02/2024
' Desenvolvido por Thiago Barros

Sub limpeza()

    Dim ws As Worksheet
    Dim ultimalinha As Long
    Dim i As Long
    Dim colunas_limpeza As String
    Dim colunas_exclusao As String
    Dim coluna As Variant
    Dim alerta As VbMsgBoxResult
    Dim precisao As Integer
    Dim referencia_exclusao As Integer

    On Error Resume Next
    Set ws = ThisWorkbook.Sheets(InputBox("Digite o nome da planilha para Executar a Limpeza:", "Nome Planilha"))
    On Error GoTo 0
    

    If ws Is Nothing Then
        MsgBox "A planilha especificada não foi encontrada.", vbExclamation
        Exit Sub
    End If
    

    colunas_limpeza = InputBox("Digite os nomes das colunas separados por vírgula para LIMPEZA (Ex: A, B, C):", "Colunas para Limpeza")
    colunas_exclusao = InputBox("Digite os nomes das colunas que deverão ter as linhas excluidas, separadas por vírgula (Ex: A, B, C):", "Colunas Para Exclusao")
    
    precisao = InputBox("Digite os valor de Casas Decimais :", "Arredondamento")
    referencia_exclusao = InputBox("Digite os valor de Referência para Exclusão :", "Valores de Referência de Exclusão")

    alerta = MsgBox("Tem certeza de que deseja Limpar as colunas: " & colunas_exclusao & "?", vbYesNo + vbQuestion)
    If alerta = vbNo Then Exit Sub
        

    Dim lista_limpeza() As String
    lista_limpeza = Split(colunas_limpeza, ",")
    
    Dim lista_exclusao() As String
    lista_exclusao = Split(colunas_exclusao, ",")
    
    For Each coluna In lista_limpeza

        ultimalinha = ws.Cells(ws.Rows.Count, coluna).End(xlUp).Row
        
        For i = ultimalinha To 1 Step -1

            If IsError(ws.Cells(i, coluna).Value) Then
                On Error Resume Next
                ws.Cells(i, coluna).Value = CDbl(ws.Cells(i, coluna).Value)
                On Error GoTo 0
            End If
            

            If IsNumeric(ws.Cells(i, coluna).Value) And Not IsEmpty(ws.Cells(i, coluna).Value) Then
                ws.Cells(i, coluna).Value = CDbl(Replace(ws.Cells(i, coluna).Value, ".", ","))
                ws.Cells(i, coluna).Value = Round((ws.Cells(i, coluna).Value), precisao)
            End If
        Next i
    Next coluna
    
    For Each coluna In lista_exclusao
    
        ultimalinha = ws.Cells(ws.Rows.Count, coluna).End(xlUp).Row
        
        For i = ultimalinha To 1 Step -1
                
                If ws.Cells(i, coluna).Value = 0 Then
                    ws.Rows(i).Delete
                End If

            Next i
    Next coluna
End Sub
Sub macro_limpeza()
Call limpeza
End Sub