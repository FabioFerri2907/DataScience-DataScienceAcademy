# Mini-Projeto_2 - Machine Learning em Prevendo Funcionamento Extintor Hidrostático
# Curso1-BigDataAzureMachineLearning
# Julho/2023
# Fabio Ferri 
# ---------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------
##### Problema de Negócio #####
# ---------------------------------------------------------------------------------------------------------------------------


#Usando dados reais disponíveis publicamente, seu trabalho é desenvolver um modelo
#de Machine Learning capaz de prever a eficiência de extintores de incêndio.

#O teste hidrostático extintor é um procedimento estabelecido pelas normas da ABNT
##NBR 12962/2016, que determinam que todos os extintores devem ser testados a cada cinco
#anos, com a finalidade de identificar eventuais vazamentos, além de também verificar a
#resistência do material do extintor.
#Com isso, o teste hidrostático extintor pode ser realizado em baixa e alta pressão, de
#acordo com estas normas em questão. O procedimento é realizado por profissionais técnicos
#da área e com a utilização de aparelhos específicos e apropriados para o teste, visto que eles
#devem fornecer resultados com exatidão.

# Os tipo e combust[iveis tem medidas diferentes separar na Análise

# ---------------------------------------------------------------------------------------------------------------------------
##### Instala os pacotes #####
# ---------------------------------------------------------------------------------------------------------------------------
setwd("C:/Users/FFERRI/Documents/Desenvolvimento/DataScienceAcademy/FormacaoCietistadeDados/Curso1-BigDataAzureMachineLearning/20-ProjetosComFeedback/Projetos-1-2/Projeto2-ExtintoresdeIncendio")
getwd()


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
library(gmodels)
library(class)
library(rpart)
#install.packages('knitr', dependencies = TRUE)



# ---------------------------------------------------------------------------------------------------------------------------
##### Análise Exploratória dos Dados - Limpeza dos Dados ##### 
# ---------------------------------------------------------------------------------------------------------------------------


# Importando uma worksheet para um dataframe
df_dados <- read_excel("dataset/Extinguisher_Fire.xlsx", sheet = 1)
View(df_dados)


# Variáveis e tipos de dados
str(df_dados)

# Sumários das variáveis numéricas
summary(df_dados)

# Tamanho Dataset 
dim(df_dados)

# Nomes das colunas
colnames(df_dados)


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

# Sem Dados nulos, a análise pode ser seguida em frente


# Remover NA ( Não existem Dados nulos)
#df_dados <- na.omit(df_dados)


# Remove os objetos anteriores para liberar memória RAM
rm(complete_cases)
rm(not_complete_cases)


# Convert Campos Fator 
df_dados$FUEL <-  as.numeric(factor(df_dados$FUEL))
levels(df_dados$FUEL)
summary(df_dados$FUEL)  

str(df_dados)  



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

# Variável dependente Target STATUS é Varável Qualitativa Ordinal
# A Variável Target 0 indicates the non-extinction state, 1 indicates the extinction state

# Variável FUEL 
fuel_table <- table(df_dados$FUEL)
barchart(fuel_table)

# Construindo o Barplot
barplot(fuel_table, beside = T)
barplot(fuel_table)

hist(fuel_table, labels = T,  breaks = 10, main = "Histograma de Combustiveis ")
View(fuel_table)

dt_fuel <- as.data.frame(fuel_table)
dt_fuel

dt_fuel %>% 
  mutate(percent = Freq / sum(Freq)  )
  head


# Variável independente  Distancia
range(numeric_data$DISTANCE)
# Ramge Min 10 Max 190 cm

mean(numeric_data$DISTANCE)
# média 100

median(numeric_data$DISTANCE) 
# Mediana 100

sd(numeric_data$DISTANCE)
# Desviio Padrão  54.77383

var(numeric_data$DISTANCE)
# Variancia 3000.172

hist(numeric_data$DISTANCE, breaks = 8)



# Variável independente  DISEBEL
range(numeric_data$DESIBEL)
# Ramge Min 72 Max 113 dcb

mean(numeric_data$DESIBEL)
# média 96.37914

median(numeric_data$DESIBEL) 
# Mediana 95

sd(numeric_data$DESIBEL)
# Desviio Padrão  54.77383

var(numeric_data$DESIBEL)
# Variancia 66.65247

hist(numeric_data$DESIBEL, breaks = 8)



# Matriz de Correlação
cor(numeric_data)

# Graficos em Pairs mostram se todas as variveis tem correção, exemplpo pode ser positiva , qd uma aumenta a outra aumenta, mas 
# não pode ter onde uma aumenta e outra diminiu
# Correlation Plot
pairs(numeric_data)


# Plot composto por varias variaveis 
xyplot(AIRFLOW ~  DISTANCE | SIZE , groups = STATUS, 
       data = numeric_data, auto.key = list(space="right"))

# AIRFLOW X DISTANCE  tem correlação Negativa, quanto maior a distancia menor o fluxo de ar. 
ggplot(data = df_dados, 
       aes(x = AIRFLOW, y = DISTANCE, 
           colour = as.factor(STATUS))) + geom_point()

# Sem correlação FLUXO DE AR e TAMANHO
ggplot(data = df_dados, 
       aes(x = AIRFLOW, y = SIZE, 
           colour = as.factor(STATUS))) + geom_point()


# Quanto Maior o Fluxo de AR Mais e Mais DISEBEL mais teve Sucesso
ggplot(data = df_dados, 
       aes(x = DESIBEL, y = AIRFLOW, 
           colour = as.factor(STATUS))) + geom_point()



# Quanto Maior o Fluxo de AR Mais e Mais DISEBEL mais teve Sucesso
ggplot(data = df_dados, 
       aes(x = DESIBEL, y = AIRFLOW, 
           colour = as.factor(STATUS))) + geom_point()



# Fluxo de Ar x sTATUS

AIRFLOWxSTATUSboxplot = boxplot(data = df_dados, AIRFLOW  ~ factor(STATUS) ,
                                     main = "Fluxo de Ar X Status",
                                     col.main = "red", ylab = "Fluxo de Ar", xlab = "Status")




# Seleciona Colunas 


df_dados_model <- numeric_data %>% 
  select( SIZE, FUEL, DISTANCE, DISTANCE, AIRFLOW , FREQUENCY, STATUS )

View(df_dados_model)


# Normalizar Dados  
normalizar <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

df_dados_model <- as.data.frame(lapply(df_dados_model, normalizar))

str(df_dados_model)
View(df_dados_model)


# ---------------------------------------------------------------------------------------------------------------------------
##### Modelos  ##### 
# ---------------------------------------------------------------------------------------------------------------------------


# Modelo 1 usando o KNN 

# Treinos e Teste 
split <- createDataPartition(y = df_dados_model$STATUS, p = 0.7, list = FALSE)

# Split dados
dados_treino <- df_dados_model[split,]
dados_teste <- df_dados_model[-split,]



# Verificando o numero de linhas
nrow(dados_treino)
nrow(dados_teste)



# Criando o modelo
modelo_knn_v1 <- knn(train = dados_treino, 
                     test = dados_teste,
                     cl = dados_treino$STATUS,
                     k = 21)


# Summary 1 Modelo
summary(modelo_knn_v1)



# Prever  Status
cat("Prever STATUS:\n", modelo_knn_v1, "\n")

cat("Dados Test STATUS:\n", dados_teste$STATUS, "\n")

# Confusion Matrix Table 
conf_matrix_tb <- table(Actual = dados_teste$STATUS, Predicted = modelo_knn_v1)
print("Confusion Matrix:")
print(conf_matrix_tb)

df_conf_matrix <- as.data.frame(conf_matrix_tb)
df_conf_matrix

ggplot(data = df_conf_matrix,
       mapping = aes(x = Actual,
                     y = Predicted)) +
  geom_tile(aes(fill = Freq)) +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "blue",
                      high = "red",
                      trans = "log")

accuracy_modelo_1_knn <- sum(diag(conf_matrix_tb))/length(dados_teste$STATUS)
sprintf("Accuracy: %.2f%%", accuracy_modelo_1_knn*100)
#Accuracy:  100  


# O Modelo 1 , fiz alguns testes e com a Normalizaçãop dos Dados ele ficoi 100%, at[e achei estranho, e tirei a normalização e caiu para 92%, sendo assim deixei 
# a normalização. Na documentção não fala sobre ser requisito normalização para o KNN


# Modelo 2 usando o  SVM 


# Treinos e Teste 
split <- createDataPartition(y = df_dados_model$STATUS, p = 0.7, list = FALSE)

# Split dados
dados_treino <- df_dados_model[split,]
dados_teste <- df_dados_model[-split,]



# Verificando o numero de linhas
nrow(dados_treino)
nrow(dados_teste)


# Criando o modelo
modelo_svm_v2 = svm(formula = STATUS ~ .,
                 data = dados_treino,
                 type = 'C-classification',
                 kernel = 'linear')

# Sumarizar Modelo 
print(modelo_svm_v2)

# Vazendo Previsões 
predicted_STATUS <- predict(modelo_svm_v2, dados_teste)



# Confusion Matrix
conf_matrix_tb <- table(Actual = dados_teste$STATUS, Predicted = predicted_STATUS)
print("Confusion Matrix:")
print(conf_matrix_tb)

df_conf_matrix <- as.data.frame(conf_matrix_tb)
df_conf_matrix

ggplot(data = df_conf_matrix,
       mapping = aes(x = Actual,
                     y = Predicted)) +
  geom_tile(aes(fill = Freq)) +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "blue",
                      high = "red",
                      trans = "log")

# Calculate accuracy
accuracy_modelo_2_svm <- sum(diag(conf_matrix_tb)) / sum(conf_matrix_tb)
cat("Accuracy: ", accuracy_modelo_2_svm, "\n")
#Accuracy:  0.8746177 

# Modelo 2 SVM também teve uma boa performance de 87% de acuracidade, mesmo assim a ConsusionMatrix apresentaou erros. 



# Modelo 3 usando o  R  Naive Bayes Model


# Treinos e Teste 
split <- createDataPartition(y = df_dados_model$STATUS, p = 0.7, list = FALSE)

# Split dados
dados_treino <- df_dados_model[split,]
dados_teste <- df_dados_model[-split,]



# Verificando o numero de linhas
nrow(dados_treino)
nrow(dados_teste)


# Criando o modelo
# Train the  Naive Bies
modelo_NB_v3 <- naiveBayes(STATUS ~ ., data = dados_treino, method = "class")



# Sumarizar Modelo 
print(modelo_NB_v3)


# Vazendo Previsões 
predicted_STATUS <- predict(modelo_NB_v3, dados_teste)



# Confusion Matrix
conf_matrix_tb <- table(Actual = dados_teste$STATUS, Predicted = predicted_STATUS)
print("Confusion Matrix:")
print(conf_matrix_tb)


df_conf_matrix <- as.data.frame(conf_matrix_tb)
df_conf_matrix

ggplot(data = df_conf_matrix,
       mapping = aes(x = Actual,
                     y = Predicted)) +
  geom_tile(aes(fill = Freq)) +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_gradient(low = "blue",
                      high = "red",
                      trans = "log")


# Calculate accuracy
accuracy_modelo_3_nb <- sum(diag(conf_matrix_tb)) / sum(conf_matrix_tb)
cat("Accuracy: ", accuracy_modelo_3_nb, "\n")
#Accuracy:  0.8704128

# Modelo 3   Naive Bayes Model também teve uma boa performance de 87% de acuracidade, igual ao Modelo 2 SVM, mesmo assim a ConsusionMatrix apresentaou erros. 





# gerar PDF
#knitr::stitch_rhtml('MiniProjeto_2_ML_PrevendoFuncionamentoExtintores.r')
# gerar PDF
#knitr::stitch('MiniProjeto_2_ML_PrevendoFuncionamentoExtintores.r')



# ---------------------------------------------------------------------------------------------------------------------------
##### Conclusão Final dos Modelo ML  #####
# ---------------------------------------------------------------------------------------------------------------------------

model_desc = c("Modelo","Acuracidade")
modelo_v1 = c("KNN",accuracy_modelo_1_knn)
modelo_v2 = c("SVM",accuracy_modelo_2_svm)
modelo_v3 = c("NB",accuracy_modelo_3_nb)



result_v1 <- as.data.frame(cbind(model_desc, modelo_v1))
result_v2 <- as.data.frame(cbind(model_desc, modelo_v2))
result_v3 <- as.data.frame(cbind(model_desc, modelo_v3))

df_results <- merge(result_v1, result_v2,result_v3) %>%
  merge(result_v3)

print(df_results)

#model_desc         modelo_v1         modelo_v2         modelo_v3
#Acuracidade        1                 0.878631498470948 0.878631498470948
#Modelo             KNN               SVM               NB

# Conclusão Final é modelo KNN tem 100% de Precisão, utilizando o modelo KNN + Normalização de Dados



# ---------------------------------------------------------------------------------------------------------------------------
##### Prever Dados Próprios  ##### 
# ---------------------------------------------------------------------------------------------------------------------------



df_novos_dados <- numeric_data %>% 
  select( SIZE, FUEL, DISTANCE, DISTANCE, AIRFLOW , FREQUENCY, STATUS )
df_prever <- sample(df_novos_dados)

df_prever <- df_prever[1:6]
#View(df_prever)

# Altera Distancia e Fluxo dw Ar primeia linha 
df_prever[1:1:1,5]   = 0
df_prever[1:1:1,3]   = 0
# Altera AIRFLOW 

View(df_prever)

df_prever <- as.data.frame(lapply(df_prever, normalizar))
View(df_prever)
df_prever_linha = df_prever[1:1:1,]


# Naive Baies
predicted_STATUS <- predict(modelo_NB_v3, df_prever_linha)
# Valor Previsto
print.factor(predicted_STATUS)




