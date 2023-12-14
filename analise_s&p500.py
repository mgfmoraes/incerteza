import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from datetime import datetime
from matplotlib.ticker import MultipleLocator

# Leitura do arquivo CSV
def le_arquivo_csv(file_path):
    #file_path = 'sp500_index.csv'
    dados = pd.read_csv(file_path)
    return dados

def gerar_grafico_linha(df, campo_valor, title, label_caption):
    # Gráfico de linha para o S&P500 ao longo do tempo
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df[f"{campo_valor}"], label=label_caption)
    plt.title(title)

    xtick_labels = [datetime.strptime(date, '%Y-%m-%d') for date in df['Date']]
    xtick_indices = np.linspace(0, len(xtick_labels)-1, 10).astype(int)
    xtick_labels = [xtick_labels[i] for i in xtick_indices]
    xtick_labels = [label.strftime('%Y-%m-%d') for label in xtick_labels]
    plt.xticks(xtick_indices, xtick_labels)

    plt.legend()
    plt.show()

def gerar_2grafico_linha():
    # Gráfico de linha para o S&P500 ao longo do tempo
    plt.figure(figsize=(12, 6))
    plt.plot(df_SP500['Date'], df_SP500['S&P500'], label='S&P500')
    plt.plot(df_dow['Date'], df_dow['DJIA'], label='Dow Jones')
    plt.title('S&P500 X Dow Jones')

    
    xtick_labels = [datetime.strptime(date, '%Y-%m-%d') for date in df_SP500['Date']]
    xtick_indices = np.linspace(0, len(xtick_labels)-1, 10).astype(int)
    xtick_labels = [xtick_labels[i] for i in xtick_indices]
    xtick_labels = [label.strftime('%Y-%m-%d') for label in xtick_labels]
    plt.xticks(xtick_indices, xtick_labels)
    

    plt.legend()
    plt.show()
    
def gerar_diagrama_caixa(df):
    plt.figure(figsize=(11,6 )) # Define o tamanho da figura
    sns.boxplot(x=df['S&P500']) # Cria o box plot com a coluna "S&P500"
    plt.xlabel('Fonte: Kagle com base nos dados do indice S&P500', fontsize=15) # Define o rótulo do eixo x com um tamanho de fonte maior
    plt.title('Grafico 1: Distribuição do indice S&P500', fontsize=16) # Define o título do gráfico com um tamanho de fonte maior

    # Análise estatística descritiva
    descricao_indice = df['S&P500'].describe()

    plt.text(0.7, 0.9, f'Média: {descricao_indice["mean"]:.2f}', transform=plt.gca().transAxes, fontsize=8)
    plt.text(0.7, 0.85, f'Mediana: {descricao_indice["50%"]:.2f}', transform=plt.gca().transAxes, fontsize=8)
    plt.text(0.7, 0.8, f'1º Quartil: {descricao_indice["25%"]:.2f}', transform=plt.gca().transAxes, fontsize=8)
    plt.text(0.7, 0.75, f'3º Quartil: {descricao_indice["75%"]:.2f}', transform=plt.gca().transAxes, fontsize=8)

    plt.show() # Exibe o gráfico
    print(df.describe())

    data = df
    data.head()
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year

    stats = data.groupby('Year')['S&P500'].agg(['min', 'max', 'mean', 'std', 'quantile'])
    stats.columns = ['Min', 'Max', 'Mean', 'Std', 'Q3']
    print(stats)

    df['Date'] = pd.to_datetime(df['Date'])

    # Adicionando uma nova coluna para o ano
    df['Ano'] = df['Date'].dt.year
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Ano', y='S&P500', data=df)

    plt.title('Boxplot dos Valores S&P500 Agrupados por Ano')
    plt.xlabel('Ano')
    plt.ylabel('Valor S&P500')
    plt.show()

def gerar_distribuicao_frequencia(df):
    # Histograma para distribuição do S&P500
    plt.figure(figsize=(12, 6))
    sns.histplot(df['S&P500'], bins=30, kde=True)
    plt.title('Distribuição do S&P500')
    plt.xlabel('Valor do S&P500')
    plt.ylabel('Frequência')
    plt.show()

def gerar_simulacao_monte_carlo(df):
    # Simulação de Monte Carlo
    mean_return = df['S&P500'].mean()
    std_dev = df['S&P500'].std()

    # Geração de cenários simulados
    num_simulations = 1000
    simulated_returns = np.random.normal(mean_return, std_dev, num_simulations)

    # Visualização dos cenários simulados
    plt.figure(figsize=(12, 6))
    sns.histplot(simulated_returns, bins=30, kde=True)
    plt.title('Simulação de Monte Carlo para Retornos do S&P500')
    plt.xlabel('Dias no Futuro')
    plt.ylabel('Retornos Simulados')
    plt.show()

def gerar_retorno_diario(df):
    # Calcular os retornos diários
    df['Returns'] = df['S&P500'].pct_change()

def gerar_grafico_retorno_diario(df):
    # Gráfico de linha para os retornos diários
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Returns'], label='Returns', color='green')
    plt.title('Retornos Diários do S&P500')

    xtick_labels = [datetime.strptime(date, '%Y-%m-%d') for date in df['Date']]
    xtick_indices = np.linspace(0, len(xtick_labels)-1, 10).astype(int)
    xtick_labels = [xtick_labels[i] for i in xtick_indices]
    xtick_labels = [label.strftime('%Y-%m-%d') for label in xtick_labels]
    
    plt.xticks(xtick_indices, xtick_labels)
    plt.legend()
    plt.show()

def calcular_correlacao():
    # Calcule a correlação
    correlation_dow = df_SP500['S&P500'].corr(df_dow['DJIA'])

    print(f"Correlação com Dow Jones: {correlation_dow}")

df_SP500 = le_arquivo_csv('sp500_index.csv')
df_dow = le_arquivo_csv('DJIA.csv')

calcular_correlacao()
gerar_retorno_diario(df_SP500)
gerar_grafico_retorno_diario(df_SP500)

volatility = df_SP500['Returns'].std()

print(f"Volatilidade: {volatility}")

gerar_grafico_linha(df_SP500,'S&P500' ,'Indice S&P500', 'S&P500' )
gerar_grafico_linha(df_dow,'DJIA', 'Indice Dow Jones', 'Dow Jones' )
gerar_2grafico_linha()
gerar_diagrama_caixa(df_SP500)
gerar_distribuicao_frequencia(df_SP500)


# Teste de normalidade
normality_test = norm.fit(df_SP500['S&P500'])
print(f"Teste de Normalidade: p-value = {normality_test}")

# Correlação e Covariância
correlation = df_SP500['S&P500'].corr(df_SP500['S&P500'].shift(1))
covariance = df_SP500['S&P500'].cov(df_SP500['S&P500'].shift(1))
print(f"Correlação: {correlation}")
print(f"Covariância: {covariance}")

gerar_simulacao_monte_carlo(df_SP500)
