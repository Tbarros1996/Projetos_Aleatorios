Attribute VB_Name = "Módulo1"
' Tirando a Média das Notas

Sub calculo_media()

x = 0
last_row = Cells(Rows.Count, 9).End(xlUp).Row

For i = 1 To last_row

    y = Cells(i, 9).Value()
    x = x + y
    
Next i

Media_calculo = x / last_row

MsgBox ("A média é de")
MsgBox (Media_calculo)

End Sub

