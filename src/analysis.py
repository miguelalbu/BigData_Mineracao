import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def carregar_dados(caminho):
    print(f"Tentando carregar arquivo de: {caminho}")
    df = pd.read_csv(caminho)
    return df

def mostrar_info(df):
    print("\nInformações do Dataset:")
    print(df.info())
    print("\nPrimeiras 5 linhas:")
    print(df.head())
    print("\nEstatísticas descritivas:")
    print(df.describe())
    print("\nValores nulos por coluna:")
    print(df.isnull().sum())

def gerar_graficos(df):
    sns.set(style="whitegrid")
    
    # 1. Gasto médio por tipo de assinatura
    plt.figure(figsize=(8,5))
    sns.barplot(x='Membership Type', y='Total Spend', data=df, ci=None, palette='Blues_d')
    plt.title('Gasto Médio por Tipo de Assinatura')
    plt.tight_layout()
    plt.savefig('grafico_gasto_medio_por_assinatura.png')
    plt.close()

    # 2. Avaliação média por cidade
    plt.figure(figsize=(10,5))
    sns.barplot(x='City', y='Average Rating', data=df, ci=None, palette='Purples_d')
    plt.title('Avaliação Média por Cidade')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('grafico_avaliacao_media_por_cidade.png')
    plt.close()

    # 3. Itens comprados por nível de satisfação (removendo nulos)
    df_satisfacao = df.dropna(subset=['Satisfaction Level'])
    plt.figure(figsize=(8,5))
    sns.barplot(x='Satisfaction Level', y='Items Purchased', data=df_satisfacao, ci=None, palette='Greens_d')
    plt.title('Média de Itens Comprados por Nível de Satisfação')
    plt.tight_layout()
    plt.savefig('grafico_itens_por_satisfacao.png')
    plt.close()

    # 4. Total gasto por aplicação de desconto
    plt.figure(figsize=(6,5))
    sns.barplot(x='Discount Applied', y='Total Spend', data=df, ci=None, palette='Oranges')
    plt.title('Gasto Médio com e sem Desconto Aplicado')
    plt.tight_layout()
    plt.savefig('grafico_gasto_por_desconto.png')
    plt.close()

    print("\nGráficos salvos com sucesso.")

def main():
    caminho = os.path.join('data', 'raw', 'E-commerce Customer Behavior - Sheet1.csv')
    df = carregar_dados(caminho)
    mostrar_info(df)
    gerar_graficos(df)

if __name__ == "__main__":
    main()