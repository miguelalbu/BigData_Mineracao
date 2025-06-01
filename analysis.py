import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)

from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# Estilo dos gráficos
plt.style.use('ggplot') 
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100

# ETAPA 1: Carregamento dos dados
df = pd.read_csv("data/E-commerce Customer Behavior - Sheet1.csv")
print(df.head())

# Pré-processamento e discretização
# Criando categorias a partir de variáveis contínuas
df['Gasto_Categ'] = pd.qcut(df['Total Spend'], q=3, labels=['Baixo', 'Médio', 'Alto'])
df['Itens_Categ'] = pd.qcut(df['Items Purchased'], q=3, labels=['Pouco', 'Médio', 'Muito'])
df['Avaliacao_Categ'] = pd.qcut(df['Average Rating'], q=3, labels=['Ruim', 'Regular', 'Boa'])

# Recência: separando por mediana
limiar_recencia = df['Days Since Last Purchase'].median()
df['Recencia'] = df['Days Since Last Purchase'].apply(lambda x: 'Recente' if x <= limiar_recencia else 'Antigo')

# Visualizações básicas
fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # Mudamos de 3 para 2 colunas e reduzimos a largura
sns.countplot(x='Gasto_Categ', data=df, ax=axes[0])
axes[0].set_title('Categorias de Gasto')

sns.countplot(x='Itens_Categ', data=df, ax=axes[1])
axes[1].set_title('Quantidade de Itens')

plt.tight_layout()
plt.show()

# Verificando os nomes corretos das colunas
print("\nColunas disponíveis no DataFrame:")
print(df.columns.tolist())

# ETAPA 4: Binarização (One-hot encoding)
# Ajustando os nomes das colunas numéricas para corresponder ao seu dataset
colunas_numericas = ['Customer ID', 'Total Spend', 'Items Purchased', 'Average Rating']
df_numericas = df[colunas_numericas].copy()

# Fazemos a binarização das variáveis categóricas
variaveis_categoricas = ['Gender', 'City', 'Membership Type', 'Gasto_Categ', 'Itens_Categ', 
                        'Avaliacao_Categ', 'Recencia', 'Satisfaction Level']
df_bin = pd.get_dummies(df, columns=variaveis_categoricas, drop_first=False)

# Convertemos apenas as colunas binarizadas para boolean
colunas_binarizadas = [col for col in df_bin.columns if col not in colunas_numericas]
df_bin[colunas_binarizadas] = df_bin[colunas_binarizadas].astype(bool)

# Visualização da binarização
bin_cols = [col for col in df_bin.columns if any(prefix in col for prefix in variaveis_categoricas)]

# Criando uma visualização mais clara usando heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df_bin[bin_cols].head(10), cmap='YlOrRd', cbar_kws={'label': 'Valor'})
plt.title('Visualização das Primeiras 10 Linhas das Variáveis Binarizadas')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Exibindo também em formato tabular (ATRAVÉS DO TERMINAL)
print("\nVisualização das colunas binarizadas (5 primeiras linhas):")
print(df_bin[bin_cols].head())


# ETAPA 5: PCA para redução de dimensionalidade
# Seleção apenas de variáveis numéricas
features_numericas = df_bin.select_dtypes(include=[np.number]).drop(columns=['Customer ID'])
scaler = StandardScaler()
features_norm = scaler.fit_transform(features_numericas)

pca = PCA(n_components=2)
pca_resultado = pca.fit_transform(features_norm)

plt.figure(figsize=(8,6))
plt.scatter(pca_resultado[:, 0], pca_resultado[:, 1], alpha=0.6)
plt.title('Redução de Dimensionalidade com PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.grid(True)
plt.show()

# ETAPA 6: Regras de associação com Apriori
df_bool = df_bin.loc[:, df_bin.apply(lambda col: set(col.unique()).issubset({0,1}))]

# Geração de itemsets frequentes
frequentes = apriori(df_bool, min_support=0.01, use_colnames=True)
if not frequentes.empty:
    regras = association_rules(frequentes, metric="confidence", min_threshold=0.6)
    regras['regra'] = regras['antecedents'].astype(str) + ' => ' + regras['consequents'].astype(str)

    top_lift = regras.sort_values(by='lift', ascending=False).head(8)

    plt.figure(figsize=(10,6))
    plt.barh(top_lift['regra'], top_lift['lift'], color='cornflowerblue')
    plt.xlabel('Lift')
    plt.title('Top Regras de Associação por Lift')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# ETAPA 7: Agrupamento com K-Means
modelo_kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
df_bin['Segmento'] = modelo_kmeans.fit_predict(features_norm)

# Visualização com PCA novamente
pca2 = PCA(n_components=2)
pca_clusters = pca2.fit_transform(features_norm)
df_bin['PC1'] = pca_clusters[:, 0]
df_bin['PC2'] = pca_clusters[:, 1]

plt.figure(figsize=(8,6))
sns.scatterplot(data=df_bin, x='PC1', y='PC2', hue='Segmento', palette='Set1')
plt.title('Segmentação de Clientes com K-Means')
plt.show()

# Análise detalhada de cada cluster
for cluster in range(3):
    cluster_data = df_bin[df_bin['Segmento'] == cluster]
    
    # Características dominantes
    print(f"\nCluster {cluster} características:")
    for col in features_numericas.columns:
        mean_val = cluster_data[col].mean()
        print(f"{col}: {mean_val:.2f}")

# ETAPA 8: Classificação com árvore de decisão
X = df_bin.drop(columns=['Customer ID', 'Segmento', 'PC1', 'PC2'])
y = df_bin['Segmento']

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=0)
modelo_arvore = DecisionTreeClassifier(max_depth=4, random_state=0)
modelo_arvore.fit(X_treino, y_treino)

# Avaliação
print("Relatório de Classificação - Árvore de Decisão:\n")
print(classification_report(y_teste, modelo_arvore.predict(X_teste)))

# Importância das variáveis
importancias = pd.Series(modelo_arvore.feature_importances_, index=X.columns)
importantes = importancias.sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
importantes.plot(kind='barh', color='orange')
plt.title('Top 10 Atributos Mais Relevantes - Árvore de Decisão')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# MLP
mlp = MLPClassifier(random_state=0)
mlp.fit(X_treino, y_treino)

# KNN
knn = KNeighborsClassifier()
knn.fit(X_treino, y_treino)

# Matriz de confusão e avaliação para cada modelo
modelos = {
    'Árvore': modelo_arvore,
    'MLP': mlp,
    'KNN': knn
}

# Criar uma única figura para todas as matrizes de confusão
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Matrizes de Confusão por Modelo')

for i, (nome, modelo) in enumerate(modelos.items()):
    y_pred = modelo.predict(X_teste)
    cm = confusion_matrix(y_teste, y_pred)
    
    # Plotar matriz de confusão
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axes[i])
    axes[i].set_title(f'{nome}')
    
    # Imprimir relatório de classificação
    print(f"\nRelatório de Classificação - {nome}:")
    print(classification_report(y_teste, y_pred))

plt.tight_layout()
plt.show()

# Limpar todas as figuras anteriores
plt.close('all')