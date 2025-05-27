# 📊 Projeto de Big Data e Mineração de Dados

## 🎯 Objetivo Geral

Este projeto tem como objetivo aplicar técnicas de mineração de dados com foco na **extração de regras de associação**, **clusterização** e **classificação supervisionada**, utilizando um dataset real do grupo de recomendação. O foco está na **descoberta de padrões ocultos**, **análise de relações entre atributos** e **geração de insights interpretáveis** a partir da estrutura dos dados.

---

## ❓ Perguntas de Pesquisa

1. Quais regras de associação podem ser extraídas dos dados transacionais?
2. Existem segmentos distintos de comportamento entre os usuários?
3. Quais atributos mais influenciam o agrupamento dos dados?
4. As estruturas encontradas nas regras de associação coincidem com os agrupamentos realizados?
5. Os modelos de classificação capturam os padrões descobertos anteriormente?

---

## 🔧 Pré-Processamento e Engenharia de Atributos

- Discretização de variáveis contínuas
- Binarização de atributos categóricos
- Conversão do dataset para formato transacional (cestas de itens)
- Tratamento de valores ausentes
- (Opcional) Redução de dimensionalidade via PCA
- Exploração de distribuições para identificar variáveis relevantes

---

## 🔍 Mineração de Regras de Associação

- **Algoritmos Utilizados:** Apriori, [Outro algoritmo, ex: FP-Growth]
- **Métricas analisadas:** Suporte, Confiança, Lift
- **Visualizações:** 
  - Tabelas com destaque visual das métricas
  - Gráficos de rede e barras ordenadas
- **Segmentações aplicadas:** [ex: por perfil de usuário, idade, região...]
- **Insights:** Regras interpretáveis foram extraídas e comparadas entre diferentes segmentos.

---

## 🧩 Clusterização (Segmentação Não Supervisionada)

- **Algoritmo Utilizado:** KMeans
- **Visualizações:** 
  - PCA (componentes principais)
  - Pairplots e scatterplots
  - Árvores de Decisão para identificação de atributos relevantes por cluster
- **Comparações com Regras de Associação:**
  - Análise cruzada dos agrupamentos com as regras extraídas
  - Identificação de atributos-chave recorrentes

---

## 🤖 Classificação Supervisionada (Complementar)

- **Algoritmos Utilizados:**
  - Árvore de Decisão
  - K-Nearest Neighbors (KNN)
  - Multi-Layer Perceptron (MLP)
- **Avaliação:**
  - Matriz de confusão
  - Acurácia, precisão, recall e F1-score
- **Objetivo:** Validar os padrões descobertos com técnicas supervisionadas

---

## 📈 Resultados e Discussão

- Descoberta de padrões interpretáveis relevantes para o domínio
- Agrupamentos coerentes com os segmentos definidos
- Confirmação dos padrões pelas métricas dos classificadores
- Discussão crítica sobre regras redundantes, discrepâncias e padrões inesperados

---

## 🛠 Reprodutibilidade

Este projeto pode ser executado diretamente via Google Colab ou ambiente local com as bibliotecas:

```bash
pip install pandas numpy matplotlib seaborn mlxtend scikit-learn
