# Primeiro, certifique-se de que as bibliotecas necessárias estão instaladas:
# Execute os seguintes comandos no terminal ou no ambiente de desenvolvimento:
# pip install pandas scikit-learn

# Importando as bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Carregar os dados históricos da Lotofácil
# O arquivo CSV precisa estar no mesmo diretório do código ou especificar o caminho completo
try:
    dados = pd.read_csv("historico_lotofacil.csv")
    print("Dados carregados com sucesso!")
except FileNotFoundError:
    print("Erro: Arquivo CSV não encontrado. Verifique o nome e o caminho do arquivo.")
    exit()

# Exibir as primeiras linhas do arquivo para verificar se o carregamento está correto
print(dados.head())

# Verifique se os dados têm as colunas corretas (de 1 a 15)
if len(dados.columns) < 16:
    print("Erro: O arquivo CSV não tem as colunas corretas. Certifique-se de que há 15 números por sorteio.")
    exit()

# Separar as colunas de entrada (números sorteados) e a coluna de saída (o número alvo que você quer prever)
X = dados.iloc[:, 1:16]  # Números sorteados (colunas 1 a 15)
y = dados['Número 1']  # Escolhendo uma coluna para prever, como o 'Número 1'. Você pode ajustar isso.

# Dividir os dados em conjuntos de treinamento (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Usar um classificador como Floresta Aleatória para prever o número
modelo = RandomForestClassifier()

# Treinar o modelo com os dados de treinamento
modelo.fit(X_train, y_train)

# Fazer previsões com os dados de teste
previsoes = modelo.predict(X_test)

# Avaliar a precisão do modelo com base nas previsões
precisao = accuracy_score(y_test, previsoes)
print(f"Precisão do modelo: {precisao * 100:.2f}%")

# Exibir as previsões feitas
print("Previsões: ", previsoes)
