import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
 
# Carregando o dataset inflacao, e gerenciando as variaveis 
data = pd.read_csv("inflacao.csv")
X = data.drop('referencia', axis=1)
y = data['ipca_variacao']

# Dividindo o dataset em treino e teste usando 80% para treinamento e 20% para teste 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

#Cria a arvore de desição usando regressão 
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Fazendo previsões do modelo
y_pred = model.predict(X_test)

# Avalianção do modelo usando erro quadratico e r quadrado
mse = mean_squared_error(y_test, y_pred) 
r2 = r2_score(y_test, y_pred)
print(f'Erro quadratico medio: {mse:.2f}')
print(f'R-Quadrado: {r2}')

