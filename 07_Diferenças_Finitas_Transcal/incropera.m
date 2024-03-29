% Exercicio Incropera 4.75 com e Método de Eliminação e Iterativo de Gauss
% sem geração de calor interna com distribuição de calor em chips de 18mm de largura aplicado
% em 1 camada de material sumidouro para dispersão do calor dos chip
% T chip = 85°C;
% após 22,5mm do sumidouro = 25°C;
% Distribuição de calor no sumidouro ???? (22,5mm);

Td = 85
Ks = 300
Ts = 25
%deltax = 9
%deltay = 4,5

%No Sumidouro:
%Nós vizinhos à T da superfície superior são (T=85°C): 1, 2, 3, 4, 5, 6
%Nós vizinhos à T da superfície inferior são (T=25°C): 19, 20, 21, 22, 23,
%24
%Nós vizinhos à lateral esquerda são: 1, 7, 13, 19
%Nós vizinhos à lateral direita são: 6, 12, 18, 24

for i = 1, 6 %laterais em contato com T=0°C e 85°C
    
Tn(i) == ((Tn(i+1) + (4)*(Tn(i+6) + 340 )) / (10))

for i = 7 %lateral esquerda apenas em contato com T=0°C
Tn(i) == ((Tn(i+1) + (4)*((Tn(i+6) + Tn(i-6)))) / (10))

for i = 2, 3, 4, 5 %superfície superior em contato com T=85°C
Tn(i) == ((Tn(i-1) + Tn(i+1) + (4)*(Tn(i+6) + 340 )) / (10))

for i = 8, 9, 10, 11, 12 %superfície interna
Tn(i) == (((Tn(i-1) + Tn(i+1) + (4)*((Tn(i+6) + Tn(i-6))) / (10))))
   
for i = 13 %lateral esquerda apenas em contato com T=0°C
Tn(i) == (((Tn(i-1) + (Tn(i+6) + (4)*Tn(i-6)))) / (10))

for i = 14, 15, 16, 17 %superfície interna
Tn(i) == ((Tn(i-1) + Tn(i+1) + (4)*((Tn(i+6) + Tn(i-6)))) / (10))

for i = 18 %lateral direita apenas em contato com T=0°C  
Tn(i) == (((Tn(i-1) + ((4)*(Tn(i+6)) + Tn(i-6)))) / (10))

for i = 19, 24 %laterais em contato com T=0°C e 85°C
Tn(i) == (((Tn(i-1) + (Tn(i-6)) + (100))) / (10))

for i = 20, 21, 22, 23 %superfície interna  
    
    
    
Tn(i) == (((Tn(i-1) + Tn(i+1) + (100) +  Tn(i-6)) / (10)))
if |Tn,(i) - Tv(i)| < e, paratodo i
    if Tv(i) == Tn(i)
    if yes Tn(i)
        if not for i = 1,23
                Tv(i) == Tn(i)
        end