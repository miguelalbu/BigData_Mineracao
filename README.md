# Análise de Comportamento de Clientes E-commerce

## 📊 Descrição
Projeto de mineração de dados focado na análise de comportamento de clientes de e-commerce, utilizando técnicas avançadas de descoberta de padrões.

## 🎯 Objetivos
- Descobrir padrões e estruturas ocultas nos dados
- Extrair regras de associação significativas
- Segmentar clientes através de clustering
- Validar descobertas com classificação supervisionada

## 🛠️ Tecnologias Utilizadas
- Python 3.12
- Pandas e NumPy
- Scikit-learn
- Matplotlib/Seaborn
- MLxtend

## 📊 Etapas do Projeto

### 1. Pré-processamento
```python
# Discretização de variáveis
df['Gasto_Categ'] = pd.qcut(df['Total Spend'], q=3, labels=['Baixo', 'Médio', 'Alto'])
df['Itens_Categ'] = pd.qcut(df['Items Purchased'], q=3, labels=['Pouco', 'Médio', 'Muito'])
df['Avaliacao_Categ'] = pd.qcut(df['Average Rating'], q=3, labels=['Ruim', 'Regular', 'Boa'])
```

### 2. Análise de Regras de Associação
- Algoritmo Apriori implementado
- Parâmetros: min_support=0.01, min_confidence=0.6
- Visualização das top 8 regras por lift

### 3. Segmentação (K-Means)
- 3 clusters identificados
- Visualização via PCA
- Análise detalhada por cluster

### 4. Classificação Supervisionada
- Árvore de Decisão
- MLP (Multi-Layer Perceptron)
- KNN (K-Nearest Neighbors)
- Matrizes de confusão comparativas


## 🚀 Como Executar

1. Clone o repositório:
```bash
git clone
cd BigData_Mineracao
```

2. Instale as dependências:
```bash
pip install
```

3. Execute a análise:
```bash
python analysis.py
```

## 📁 Estrutura do Projeto
```
BigData_Mineracao/
│   analysis.py
│   README.md
│   
└───data
        E-commerce Customer Behavior - Sheet1.csv

```

## 📝 Notas
- Dataset contém informações de comportamento de clientes
- Foco em descoberta de padrões e não em previsão
- Resultados validados por múltiplas técnicas

---
Projeto desenvolvido para a disciplina de Big Data e Mineração de Dados