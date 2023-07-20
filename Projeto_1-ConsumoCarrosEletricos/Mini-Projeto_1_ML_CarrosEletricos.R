# Mini-Projeto_1 - Machine Learning em Logística Prevendo o Consumo de Energia de Carros Elétricos
# Curso1-BigDataAzureMachineLearning
# Julho/2023
# Fabio Ferri 
# ---------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------
##### Problema de Negócio #####
# ---------------------------------------------------------------------------------------------------------------------------

#Uma empresa da área de transporte e logística deseja migrar sua frota para carros elétricos com o objetivo de reduzir os custos.
#Antes de tomar a decisão, a empresa gostaria de prever o consumo de energia de carros elétricos com base em diversos fatores de utilização e características dos veículos.

# Usando um incrível dataset com dados reais disponíveis publicamente, você deverá construir um modelo de Machine Learning capaz de prever o consumo de energia de carros elétricos com base em diversos fatores, tais como o tipo e número de motores elétricos do veículo, o peso do veículo, a capacidade de carga, entre outros atributos.

# Data Set 
#https://data.mendeley.com/datasets/tb9yrptydn/2


# ---------------------------------------------------------------------------------------------------------------------------
##### Instala os pacotes #####
# ---------------------------------------------------------------------------------------------------------------------------
setwd("C:/Users/FFERRI/Documents/Desenvolvimento/DataScienceAcademy/FormacaoCietistadeDados/Curso1-BigDataAzureMachineLearning/20-ProjetosComFeedback/Projetos-1-2/Projeto_1-ConsumoCarrosEletricos")
getwd()


#install.packages("LiblineaR")
#0stall.packages("Metrics")

library("caret")
library("readxl")
library("corrplot")
library("dplyr")
library("ggplot2")
library("randomForest")
library("varImp")
library("readxl")
library(e1071)
library(Metrics)


# ---------------------------------------------------------------------------------------------------------------------------
##### Análise Exploratória dos Dados - Limpeza dos Dados ##### 
# ---------------------------------------------------------------------------------------------------------------------------

# Importando uma worksheet para um dataframe
df_dados <- read_excel("FEV-data-Excel.xlsx", sheet = 1)
View(df_dados)


# Variáveis e tipos de dados
str(df_dados)

# Sumários das variáveis numéricas
summary(df_dados)

# Nomes das colunas
colnames(df_dados)

# Grava os nomes das colunas em um vetor
myColumns <- colnames(df_dados)
myColumns

# Vamos renomar as colunas para facilitar nosso trabalho mais tarde
myColumns[1] <- "CarroNome"                         
myColumns[2] <-"Fabricante"                                  
myColumns[3] <-"Modelo"                                 
myColumns[4] <-"PrecoMIn"           
myColumns[5] <-"MotorKM"                     
myColumns[6] <-"TorqueMax"                   
myColumns[7] <-"TiposFreios"                        
myColumns[8] <-"Cambio"                            
myColumns[9] <-"CapacidadeBateria"                
myColumns[10] <-"Autonomia"                     
myColumns[11] <-"DistanciaEixosCM"                        
myColumns[12] <-"TamCM"                           
myColumns[13] <-"LarguraCM"                            
myColumns[14] <-"ComprimentoCM"                           
myColumns[15] <-"PesoVazio"             
myColumns[16] <-"PesoMax"         
myColumns[17] <-"CapacidadeMaxima"            
myColumns[18] <-"NumBancos"                       
myColumns[19] <-"NumPortas"                       
myColumns[20] <-"TamPneu"                        
myColumns[21] <-"VelocidadeMax"                   
myColumns[22] <-"CapacidadeIni"               
myColumns[23] <-"Acceleracao0100"            
myColumns[24] <-"Torque"        
myColumns[25] <-"MediaConsumoEnergia"
myColumns[26] <-"Make_fac"    


# Verifica o resultado
myColumns

# Atribui os novos nomes de colunas ao dataframe
colnames(df_dados) <- myColumns
rm(myColumns)
View(df_dados)

# Verificando ocorrência de valores NA
colSums(is.na(df_dados)) 
sum(is.na(df_dados))
any(is.na(df_dados))

#df_dados <- na.omit(df_dados)
sum(is.na(df_dados))

# Quantas linhas tem casos completos?
complete_cases <- sum(complete.cases(df_dados))

# Quantas linhas tem casos incompletos?
not_complete_cases <- sum(!complete.cases(df_dados))

# Qual o percentual de dados incompletos?
percentual <- (not_complete_cases / complete_cases) * 100
percentual

# Remover NA
df_dados <- na.omit(df_dados)

# Remove os objetos anteriores para liberar memória RAM
rm(complete_cases)
rm(not_complete_cases)


# Nomes das colunas
colnames(df_dados)

# Verificando os níveis do fator. Perceba que os níveis estão categorizados em ordem alfabética
levels(df_dados$Fabricante)


# Extraindo as variáveis numéricas

numeric_vars <- sapply(df_dados, is.numeric)
numeric_data <- df_dados[, numeric_vars]
str(numeric_data)

# Extraindo as variáveis  categóricas
categorical_vars <- !numeric_vars
categorical_data <- df_dados[, categorical_vars]
str(categorical_data)

# Medindo as amplitudes-----------------------------
summary(numeric_data)

# Variável dependente Target MediaConsumoEnergia
range(numeric_data$MediaConsumoEnergia)
# Ramge Min Consumo 13.10km  27.55km

mean(numeric_data$MediaConsumoEnergia)
# média 18.61071

median(numeric_data$MediaConsumoEnergia) 
# Mediana 16.875

sd(numeric_data$MediaConsumoEnergia)
# Desviio Padrão 4.134293

var(numeric_data$MediaConsumoEnergia)
# Variancia 17.09238


# Média de Autonomia 351.7381 KM
mean(numeric_data$Autonomia)

# Média de Preço é 235065.8 235.000 
mean(df_dados$PrecoMIn)

# Mediana Preço Min 166.945
median(df_dados$PrecoMIn)

# Min e Max PreçoMIn 82050 794000
range(df_dados$PrecoMIn)


# Convert Campos Fator 
df_dados$Fabricante <- as.factor(df_dados$Fabricante)
levels(df_dados$Fabricante)
summary(df_dados$Fabricante)

df_dados$Modelo <- as.factor(df_dados$Modelo)
levels(df_dados$Modelo)
summary(df_dados$Modelo)

# Variáveis e tipos de dados
str(df_dados)


# Matriz de Correlação
cor(numeric_data)

# Não foram encontrado valores outliers

# Visualizando os dados


quantile(df_dados$MediaConsumoEnergia)
#13.1000 15.6000 16.8750 22.9375 27.5500

# Fabricante
# Audi , Kia, Porche e Volksvagen possui mais carros eletricos
gbar <- ggplot(df_dados, aes(Fabricante))
gbar + geom_bar()

# Histograma consumos energéticos dos carro
# Percebe-se que o pico fica em 16, 17 na média de consumo 
hist(df_dados$MediaConsumoEnergia)

# MediaConsumoEnergia
gboxplot <- ggplot(df_dados, aes(MediaConsumoEnergia))
gboxplot + geom_boxplot()

#  MediaConsumoEnergia X Capacidade Bateria
ConsumoXBateria = boxplot(data = df_dados, MediaConsumoEnergia ~ CapacidadeBateria,
                       main = "MediaConsumoEnergia X Capacidade Bateria",
                       col.main = "red", ylab = "MediaConsumoEnergia", xlab = "capacidadeBateria")




#  Capacidade Bateria X Autonomia X Fabricante
# Audi, BMW, Citroen, DS, Honda e Hynday são os fabricantes que mais tem autonomia de uso acima de 400KM
ggplot(df_dados, aes(x = CapacidadeBateria, y = Autonomia, colour = Fabricante)) + 
  geom_point()
 
# Capacidade de Bateria X Autonomia 
# Quanto maior a capacidade de bateria maior a autonomia
ggplot(df_dados, aes(x = CapacidadeBateria, y = Autonomia, colour = Modelo)) + 
  geom_point()

# Maior parte dos veículos tem autonomia acima de 200 Km e  50O KM , 
# custam até 400.000
ggplot(df_dados, aes(x = PrecoMIn, y = Autonomia, colour = Modelo)) + 
  geom_point()



# Agrupando os dados e calculando média  Energy consumption [kWh/100 km] 

# group_by()
df_dados %>% 
  group_by(Modelo) %>%
  summarise(avg_MediaConsumoEnergia = mean(MediaConsumoEnergia), 
            min_MediaConsumoEnergia = min(MediaConsumoEnergia), 
            max_MediaConsumoEnergia = max(MediaConsumoEnergia),
            total = n())

# Carros Unicos
carros = df_dados$CarroNome
unique(carros) 

# Modelos 
Fabricante = df_dados$Fabricante
unique(Fabricante)  


# Carro mais economico
Carro_min <- df_dados %>%
  group_by(Modelo) %>%
  summarise(min_value = min(MediaConsumoEnergia))


# Carro mais consumo 
Carro_max <- df_dados %>%
  group_by(Modelo) %>%
  summarise(max_value = max(MediaConsumoEnergia))
Carro_max

# Consumo de MediaConsumoEnergia por Fabricante
MediaConsumoEnergiaboxplot = boxplot(data = df_dados, MediaConsumoEnergia  ~ factor(Fabricante) ,
                                     main = "MediaConsumoEnergia",
                                     col.main = "red", ylab = "MediaConsumoEnergia", xlab = "Fabricante")





# randomForest para encontrar as variáveis mais relevantes
regressor <- randomForest( MediaConsumoEnergia ~ . , data = numeric_data, importance=TRUE) # fit the random forest with default parameter
caret::varImp(regressor)


# Depois de executar o Randomforest, encontramos as variaveis mais relevantes a variaval target
##################################  OverallPrecoMIn    7.503244
#MotorKM            4.976540
#TorqueMax          5.546578
#CapacidadeBateria  4.319913
#Autonomia          2.743444
##################################  DistanciaEixosCM  10.847786
##################################  TamCM              8.913345
##################################  LarguraCM          5.956610
#ComprimentoCM      1.857416
##################################  PesoVazio          6.816251
##################################  PesoMax            9.963957
##################################  CapacidadeMaxima   6.496824
#NumBancos          2.702918
#NumPortas         -1.246418
##################################  TamPneu            5.077130
#VelocidadeMax      4.324391
#CapacidadeIni      4.305913
#Acceleracao0100    2.326368
#Torque             2.838302

# Seleciona Colunas

df_dados_model <- numeric_data %>% 
  select( PrecoMIn, DistanciaEixosCM , DistanciaEixosCM,TamCM,LarguraCM,PesoVazio,PesoMax,CapacidadeMaxima,TamPneu, MediaConsumoEnergia)
  
View(df_dados_model)
  

# Normalizar Dados  
normalizar <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

df_dados_norm <- as.data.frame(lapply(df_dados_model, normalizar))

str(df_dados_norm)
View(df_dados_norm)


# ---------------------------------------------------------------------------------------------------------------------------
##### Modelos  ##### 
# ---------------------------------------------------------------------------------------------------------------------------


# Modelo 1 

# Treinos e Teste 
split <- createDataPartition(y = df_dados_model$MediaConsumoEnergia, p = 0.7, list = FALSE)

# Split dados
dados_treino <- df_dados_model[split,]
dados_teste <- df_dados_model[-split,]

# Treino 1 modelo
modelo_v1 <- lm(MediaConsumoEnergia ~ ., data = dados_treino)


# Summary 1 Modelo
summary(modelo_v1)


# Plot Modelo 1
plot(modelo_v1)


#colocar o gráfico de curcva de erro

# Importancia de Variável 
#varimp(modelo_v1)
vimp  <- caret::varImp(modelo_v1)
plot(vimp)


#PrecoMIn         3.54113987
#DistanciaEixosCM 0.18599072
#TamCM            1.37321574
#LarguraCM        1.37760304
#PesoVazio        1.68842605
#PesoMax          0.13567274
#CapacidadeMaxima 0.07599211
#TamPneu          3.43723349


# Primeiro modelo r-squared 0.92 92% e p-value de 2.472e-10 baixo. 
# O modelo 1 para esta com um bom desempenho de 92%
# Variáveis mais relevantes são PrecoMIn,TamPneu, PesoVazio e LarguraCM



# Previsoes
?predict
predictedValues <- predict(modelo_v1, dados_teste)
predictedValues

# Calcula Acuracidade
modelo_v1_mse   <- MSE(dados_teste$MediaConsumoEnergia, predictedValues)
modelo_v1_rmse  <- RMSE(dados_teste$MediaConsumoEnergia,predictedValues)
modelo_v1_mae   <- MAE(dados_teste$MediaConsumoEnergia, predictedValues)
modelo_v1_r2    <- R2(dados_teste$MediaConsumoEnergia, predictedValues, form = "traditional")

#model_desc         modelo_v1
#1        MSE  2.29984730594314
#2       RMSE    1.516524746235
#3        MAE  1.14608042292544
#4         R2 0.865442359399829

# Plot Teste X Previsto
ggplot(data = dados_teste , aes(x = MediaConsumoEnergia , y = predictedValues)) + 
  ggtitle("Plot Comparativo  Teste X Previsto Modelo 1") +
  geom_point(shape = 1) +  
  geom_smooth(method = lm , color = "red", se = FALSE)  +
    xlab("Valores Previstos") + ylab("Dados Testes")



# Modelo 2

# Mostrando a importância das variáveis para a criação do modelo 1 

# Seleciona Colunas Modelo 2 somente com as variávies com maior correlação do Modelo 1 

df_dados_model2 <- numeric_data %>% 
  select( PrecoMIn, TamCM, LarguraCM,PesoVazio,MediaConsumoEnergia)

View(df_dados_model2)

# Treinos e Teste 
split <- createDataPartition(y = df_dados_model2$MediaConsumoEnergia, p = 0.7, list = FALSE)

# Split dados
dados_treino <- df_dados_model2[split,]
dados_teste <- df_dados_model2[-split,]

# Treino 2 modelo
modelo_v2 <- lm(MediaConsumoEnergia ~ ., data = dados_treino)


# Summary 2 Modelo
summary(modelo_v2)

# Plot Modelo 2
plot(modelo_v2)



# Importancia de Variável 
vimp  <- caret::varImp(modelo_v2)
plot(vimp)

# Segundo  modelo r-squared 0.40 92% e p-value de 2.472e-10 baixo. 
# O modelo 1 para esta com um bom desempenho de 40%
# Variáveis mais relevantes são PrecoMIn,TamPneu, PesoVazio e LarguraCM


# Previsoes
?predict
predictedValues <- predict(modelo_v2, dados_teste)
predictedValues

# Calcula Acuracidade
modelo_v2_mse   <- MSE(dados_teste$MediaConsumoEnergia, predictedValues) 
modelo_v2_rmse   <- RMSE(dados_teste$MediaConsumoEnergia,predictedValues)
modelo_v2_mae   <- MAE(dados_teste$MediaConsumoEnergia, predictedValues)
modelo_v2_r2    <- R2(dados_teste$MediaConsumoEnergia, predictedValues, form = "traditional")

#model_desc         modelo_v2
#1        MSE  2.74418698128479
#2       RMSE  1.65655877688804
#3        MAE  1.43847947829882
#4         R2 0.743557215863707

# Plot Teste X Previsto
ggplot(data = dados_teste , aes(x = MediaConsumoEnergia , y = predictedValues)) + 
  ggtitle("Plot Comparativo  Teste X Previsto Modelo 2") +
  geom_point(shape = 1) +  
  geom_smooth(method = lm , color = "red", se = FALSE)  +
  xlab("Valores Previstos") + ylab("Dados Testes")

# ---------------------------------------------------------------------------------------------------------------------------
##### Conclusão Final dos Modelo ML  #####
# ---------------------------------------------------------------------------------------------------------------------------
model_desc = c("MSE","RMSE","MAE","R2")
modelo_v1 = c(modelo_v1_mse,modelo_v1_rmse,modelo_v1_mae, modelo_v1_r2)
modelo_v2 = c(modelo_v2_mse,modelo_v2_rmse,modelo_v2_mae, modelo_v2_r2)


result_v1 <- as.data.frame(cbind(model_desc, modelo_v1))
result_v2 <- as.data.frame(cbind(model_desc, modelo_v2))

# Modelo 1 tem menor MSE error e tem maior R2, logo o Modelo 1 tem melhor desempenho que modelo 2 
result_v1
result_v2


#model_desc         modelo_v1
#1        MSE  2.29984730594314
#2       RMSE    1.516524746235
#3        MAE  1.14608042292544
#4         R2 0.865442359399829
#> result_v2
#model_desc         modelo_v2
#1        MSE  2.74418698128479
#2       RMSE  1.65655877688804
#3        MAE  1.43847947829882
#4         R2 0.743557215863707






















