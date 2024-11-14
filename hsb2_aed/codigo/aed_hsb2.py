import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway
import os

sns.set(style="whitegrid")

df = pd.read_csv('dados/hsb2n.csv', sep=';')

df.columns = ['id', 'genero', 'raca', 'classe_social', 'tipo_escola', 'programa_ensino', 
              'pontuacao_leitura', 'pontuacao_escrita', 'pontuacao_matematica', 'pontuacao_ciencias', 'pontuacao_estudos_sociais']

variaveis_categoricas = ['genero', 'raca', 'classe_social', 'tipo_escola', 'programa_ensino']
for var in variaveis_categoricas:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=var)
    plt.title(f'Distribuição da variável {var}')
    plt.savefig(f'figuras/dist_{var}.png')
    plt.show()

variaveis_continuas = ['pontuacao_leitura', 'pontuacao_escrita', 'pontuacao_matematica', 'pontuacao_ciencias', 'pontuacao_estudos_sociais']
for var in variaveis_continuas:
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df[var], kde=True)
    plt.title(f'Histograma de {var}')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df[var])
    plt.title(f'Boxplot de {var}')
    
    plt.savefig(f'figuras/hist_box_{var}.png')
    plt.show()

plt.figure(figsize=(10, 8))
matriz_correlacao = df[variaveis_continuas].corr()
sns.heatmap(matriz_correlacao, annot=True, cmap="coolwarm", center=0)
plt.title("Matriz de Correlação das Variáveis Contínuas")
plt.savefig('figuras/matriz_correlacao.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="genero", y="pontuacao_matematica")
plt.title("Comparação do Desempenho em Matemática por Gênero")
plt.savefig('figuras/boxplot_genero_matematica.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="raca", y="pontuacao_leitura")
plt.title("Comparação do Desempenho em Leitura por Raça")
plt.savefig('figuras/boxplot_raca_leitura.png')
plt.show()

pontuacao_leitura_masculino = df[df['genero'] == 0]['pontuacao_leitura']
pontuacao_leitura_feminino = df[df['genero'] == 1]['pontuacao_leitura']
t_stat, p_valor = ttest_ind(pontuacao_leitura_masculino, pontuacao_leitura_feminino)
print(f"Teste t para Leitura por Gênero: t-statistic = {t_stat}, p-value = {p_valor}")

grupo_raca_1 = df[df['raca'] == 1]['pontuacao_leitura']
grupo_raca_2 = df[df['raca'] == 2]['pontuacao_leitura']
grupo_raca_3 = df[df['raca'] == 3]['pontuacao_leitura']
grupo_raca_4 = df[df['raca'] == 4]['pontuacao_leitura']
anova_stat, anova_p_valor = f_oneway(grupo_raca_1, grupo_raca_2, grupo_raca_3, grupo_raca_4)
print(f"ANOVA para Leitura por Raça: F-statistic = {anova_stat}, p-value = {anova_p_valor}")
