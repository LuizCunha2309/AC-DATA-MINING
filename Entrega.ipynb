import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Carregando a base de dados
df = pd.read_csv('https://raw.githubusercontent.com/LuizCunha2309/AC-DATA-MINING/main/eth_price/eth_price.csv', thousands=',').replace(
    {'Price': '[$,]'}, {'Price': ''}, regex=True
    ).astype({'Price': 'float'})


# Visualizando as primeiras linhas
print(df.head())


# Visualizando informações sobre as colunas
print(df.info())


# Descrevendo as estatísticas básicas das variáveis numéricas
print(df.describe())


# Verificando a correlação entre as variáveis
print(df.corr(numeric_only=True))

# Separando os dados em treino e teste
X = df[['Open', 'High', 'Low']]  # Variáveis independentes
y = df['Price']  # Variável dependente
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
  )


# Treinando o modelo
model = LinearRegression()
model.fit(X_train, y_train)


# Fazendo previsões para os dados de teste
y_pred = model.predict(X_test)


# Métricas de qualidade
mse = mean_squared_error(y_test, y_pred)  # cálculo do erro quadrático médio
# MSE mede a média dos erros quadráticos
# entre os valores previstos e os valores reais

r2 = r2_score(y_test, y_pred)  # cálculo do R²
# R² é uma medida que indica a proporção da
# variância da variável dependente explicada pelo modelo


print(f'MSE: {mse:.2f}\nR2: {r2:.2f}')
# exibição dos resultados das métricas de
# qualidade para avaliar o desempenho do modelo.
