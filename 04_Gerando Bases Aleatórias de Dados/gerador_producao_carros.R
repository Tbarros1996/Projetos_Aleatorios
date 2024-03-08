library(dplyr)

set.seed(123)  # Define uma semente para reproduzibilidade

# Criação da base de dados
n <- 100000  # Número de registros

# Gerar dados aleatórios
data <- data.frame(
  id_do_produto = 1:n,
  país_de_venda = sample(c("Estados Unidos", "Canadá", "México", "Argentina", "Brasil", "Chile", "Colômbia", "Equador", "Paraguai", "Peru", "Uruguai", "Venezuela"), n, replace = TRUE),
  quantidade_vendida = sample(6000:60000, n, replace = TRUE),
  custos_operacionais = runif(n, min = 1000, max = 5000),
  mes = sample(1:12, n, replace = TRUE),
  ano = sample(2020:2024, n, replace = TRUE),
  lucro_total = numeric(n)
)

# Definir lucro total proporcional à quantidade vendida
data$lucro_total <- data$quantidade_vendida * runif(n, min = 0.05, max = 0.2)

# Definir id da marca
data$id_marca <- sample(1:7, n, replace = TRUE)

# Mostrar as primeiras linhas da base de dados
head(data)
