# Algoritmo para Gerar Base de Dados Aleatória de Produção em Relação a Consumo de Um Consumível

"""
Aplicação: Esse algoritmo (Todos os Créditos ao Chat GPT ;), cria uma base de dados onde
um conjunto de Linhas de produção estão normalizadas e relacionadas a uma variável
Consumo em M3. 

Ex: Uma padaria possui diferentes tipos de pão, e diariamente(ou por hora ou semanalmente)
tem consumo diferentes de gás ou qualquer tipo de utilidade para manter essa produção.

Qual a relação entre as variáveis.

A idéia dessa base de dados é criar uma base para estudos estatísticos e criação de modelos
de regressão.

"""


# Definindo o número de dias (2 anos * 365 dias)
num_dias <- 4 * 365

# Inicializando a base de dados
dados <- data.frame(matrix(0, nrow = num_dias, ncol = 12))

# Definindo nomes das colunas
colnames(dados) <- c(paste0("Pão_", 1:10), "Gas_Consumido_em_M3")

# Definindo pesos para cada linha de produção
pesos_linhas <- c(0.1, 0.05, 0.15, 0.2, 0.1, 0.1, 0.05, 0.05, 0.1, 0.05, 0.05)

# Definindo o máximo de gás consumido
max_gas_consumido <- 800

# Loop para preencher os dados para cada dia
for (i in 1:num_dias) {
  # Gerando um número racional para representar o consumo de gás
  gas_consumido <- runif(1, 0, max_gas_consumido)
  
  # Verificando se o dia atual é um dia em que a produção deve ser zerada
  if (runif(1) < 0.03) { # Probabilidade de 3% de zerar a produção em um dia aleatório
    dados[i, 1:11] <- 0
    dados[i, 12] <- runif(1, 78, 90)
  } else {
    # Gerando produção proporcional ao consumo de gás
    proporcoes <- pesos_linhas * runif(11) # Usando os pesos das linhas
    proporcoes <- proporcoes / sum(proporcoes) # Normalizando as proporções para somarem 1
    producao <- round(proporcoes * gas_consumido) # Arredondando para números inteiros
    
    # Preenchendo os dados de produção e consumo de gás para o dia atual
    dados[i, 1:11] <- producao
    dados[i, 12] <- gas_consumido
  }
}

# Exportando os dados para um arquivo CSV
write.csv2(dados, file = "dados_producao.csv", row.names = FALSE)

