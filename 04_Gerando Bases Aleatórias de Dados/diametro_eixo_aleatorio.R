# Gerando uma Base de Dados de Diâmetros

media <- 100
sd <-  2.5
df <- rnorm(60,mean=media,sd=sd)
write.csv2(df,file="eixo.csv")