import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Carregando a base de dados
df = pd.read_csv('https://raw.githubusercontent.com/LuizCunha2309/AC-DATA-MINING/main/eth_price/eth_price.csv', thousands=',').replace(
    {'Price': '[$,]'}, {'Price': ''}, regex=True
).astype({'Price': 'float'})

# Definindo os limiares para classificação
threshold_low = 100  # Limiar para atribuir classe 0
threshold_high = 1000  # Limiar para atribuir classe 1

# Atribuindo classes com base nos limiares definidos
df['Class'] = pd.cut(df['Price'], bins=[float('-inf'), threshold_low, float('inf')], labels=[0, 1])

# Separando os dados em treino e teste
X_linear = df[['Open']]  # Variável independente para regressão linear
y_linear = df['Price']  # Variável dependente para regressão linear
X_logistic = df[['Open', 'High', 'Low']]  # Variáveis independentes para regressão logística
y_logistic = df['Class']  # Variável dependente para regressão logística

# Regressão Linear Simples
X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(
    X_linear, y_linear, test_size=0.2, random_state=42
)
linear_model = LinearRegression()
linear_model.fit(X_train_linear, y_train_linear)
y_pred_linear = linear_model.predict(X_test_linear)
mse_linear = mean_squared_error(y_test_linear, y_pred_linear)
r2_linear = r2_score(y_test_linear, y_pred_linear)

# Regressão Linear Múltipla
X_train_multiple, X_test_multiple, y_train_multiple, y_test_multiple = train_test_split(
    X_logistic, y_logistic, test_size=0.2, random_state=42
)
multiple_model = LinearRegression()
multiple_model.fit(X_train_multiple, y_train_multiple)
y_pred_multiple = multiple_model.predict(X_test_multiple)
mse_multiple = mean_squared_error(y_test_multiple, y_pred_multiple)
r2_multiple = r2_score(y_test_multiple, y_pred_multiple)

# Regressão Logística Simples
X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = train_test_split(
    X_linear, y_logistic, test_size=0.2, random_state=42
)
logistic_model = LogisticRegression()
logistic_model.fit(X_train_logistic, y_train_logistic)
y_pred_logistic = logistic_model.predict(X_test_logistic)
accuracy_logistic = accuracy_score(y_test_logistic, y_pred_logistic)

# Regressão Logística Múltipla
X_train_logistic_multiple, X_test_logistic_multiple, y_train_logistic_multiple, y_test_logistic_multiple = train_test_split(
    X_logistic, y_logistic, test_size=0.2, random_state=42
)
logistic_multiple_model = LogisticRegression()
logistic_multiple_model.fit(X_train_logistic_multiple, y_train_logistic_multiple)
y_pred_logistic_multiple = logistic_multiple_model.predict(X_test_logistic_multiple)
accuracy_logistic_multiple = accuracy_score(y_test_logistic_multiple, y_pred_logistic_multiple)

# Resultados
print("Regressão Linear Simples:")
print(f'MSE: {mse_linear:.2f}\nR2: {r2_linear:.2f}\n')

print("Regressão Linear Múltipla:")
print(f'MSE: {mse_multiple:.2f}\nR2: {r2_multiple:.2f}\n')

print("Regressão Logística Simples:")
print(f'Accuracy: {accuracy_logistic:.2f}\n')

print("Regressão Logística Múltipla:")
print(f'Accuracy: {accuracy_logistic_multiple:.2f}\n')
