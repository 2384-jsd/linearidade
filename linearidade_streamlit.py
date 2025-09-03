# -*- coding: utf-8 -*-
"""
Aplicativo Streamlit para Análise de Linearidade na Validação Analítica
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import streamlit as st
from scipy.stats import f, pearsonr
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.stats.anova import anova_lm

# Configurações
plt.style.use('default')
st.set_page_config(page_title="Linearidade - Validação Analítica", layout="wide")
pd.options.display.float_format = '{:.4f}'.format

# Título da aplicação
st.title('Linearidade para Validação Analítica')

st.write('''#### Realizar testes estatísticos da linearidade da Validação Analítica de acordo com a RDC 166.

Este é um requisito para os métodos que foram desenvolvidos e estão sendo utilizados para liberação de resultados.
''')

# Sidebar para entrada de dados
st.sidebar.header("Configurações dos Dados")
use_example_data = st.sidebar.checkbox("Usar dados de exemplo", value=True)

if not use_example_data:
    st.sidebar.info("Funcionalidade de entrada personalizada será implementada em versão futura")

# Dados de exemplo
dados = [
    [0.157043,4660341,1,14],
    [0.160399,4754809,1,12],
    [0.160879,4681500,1,2],
    [0.176673,5384404,2,7],
    [0.180449,5428730,2,4],
    [0.180989,5263780,2,15],
    [0.196304,5843775,3,10],
    [0.200499,5993295,3,1],
    [0.201099,5841594,3,11],
    [0.215934,6555505,4,8],
    [0.220549,6607388,4,6],
    [0.221209,6294577,4,3],
    [0.235564,7109287,5,9],
    [0.240599,7181692,5,5],
    [0.241318,7095422,5,13],
]

# Criar DataFrame
df_cabeçalho = pd.DataFrame(dados, columns=['Concentração mg/mL', 'Área', 'Nível', 'Ordem de Coleta'])

st.header("1 - Introdução")
st.write("""
A linearidade de um procedimento analítico é a sua capacidade de obter resultados que sejam diretamente proporcionais à concentração de um analito em uma amostra.
""")

st.header("2 - Coleta de Dados")
st.write("A seguir, apresentam-se os dados coletados:")
st.subheader("Tabela 1 - Conjunto de dados para o estudo de Linearidade")
st.dataframe(df_cabeçalho)

st.header("3 - Método dos Mínimos Quadrados Ordinários")
st.write("""
O Método dos Mínimos Quadrados é uma eficiente estratégia de estimação dos parâmetros da regressão e sua aplicação não é limitada apenas às relações lineares.
""")

st.subheader("3.1 - Teste do Coeficiente Angular")
st.write("""
Para avaliar a significância do modelo utilizamos o teste F da ANOVA. Testa-se as hipóteses:

**H0:** coeficiente angular igual a zero  
**H1:** coeficiente angular diferente de zero
""")

# Ajustar modelo de regressão linear
model_sm = ols('Área ~ Q("Concentração mg/mL")', data=df_cabeçalho).fit()

# ANOVA
anova_table = anova_lm(model_sm)
anova_summary = anova_table.loc[['Q("Concentração mg/mL")', 'Residual'],
                                ['df', 'sum_sq', 'mean_sq', 'F', 'PR(>F)']]
anova_summary.columns = ['Graus de Liberdade', 'Soma dos Quadrados', 'Quadrado Médio', 'Estatística F', 'P-valor']

st.subheader("Tabela 2 - ANOVA")
st.dataframe(anova_summary)

# Interpretação ANOVA
alpha = 0.05
p_value_concentration = anova_summary.loc['Q("Concentração mg/mL")', 'P-valor']

st.subheader("Interpretação do Teste F da ANOVA:")
if p_value_concentration < alpha:
    st.success(f"P-valor ({p_value_concentration:.4f}) < {alpha}: Rejeitamos H0. O coeficiente angular é estatisticamente significativo.")
else:
    st.warning(f"P-valor ({p_value_concentration:.4f}) ≥ {alpha}: Não rejeitamos H0. Coeficiente angular não é estatisticamente significativo.")

# Coeficientes
st.subheader("Tabela 3 - Coeficientes")
results_summary = model_sm.summary2().tables[1]
coefficients_table = results_summary.loc[['Intercept', 'Q("Concentração mg/mL")'],
                                         ['Coef.', 'Std.Err.', 't', 'P>|t|']]
coefficients_table.columns = ['Estimativa', 'Desvio Padrão', 'Estatística t', 'P-valor']
st.dataframe(coefficients_table)

# Intervalos de confiança
st.subheader("Tabela 4 - Intervalos de Confiança")
confidence_interval_table = results_summary.loc[['Intercept', 'Q("Concentração mg/mL")'],
                                                ['Coef.', '[0.025', '0.975]']]
confidence_interval_table.columns = ['Estimativa', 'IC 95% Inferior', 'IC 95% Superior']
st.dataframe(confidence_interval_table)

# Teste do intercepto
st.subheader("3.2 - Teste do Intercepto")
intercept_p_value = coefficients_table.loc['Intercept', 'P-valor']
st.write(f"**P-valor para o Intercepto:** {intercept_p_value:.4f}")

if intercept_p_value > alpha:
    st.info(f"P-valor ({intercept_p_value:.4f}) > {alpha}: Não rejeitamos H0. O intercepto não é estatisticamente diferente de zero.")
else:
    st.success(f"P-valor ({intercept_p_value:.4f}) ≤ {alpha}: Rejeitamos H0. O intercepto é estatisticamente diferente de zero.")

# Qualidade do ajuste
st.subheader("3.3 - Qualidade do Ajuste")

X = df_cabeçalho[['Concentração mg/mL']]
y = df_cabeçalho['Área']
model_sklearn = LinearRegression()
model_sklearn.fit(X, y)

residuals = y - model_sklearn.predict(X)
r_squared = model_sklearn.score(X, y)
correlation_coefficient, _ = pearsonr(df_cabeçalho['Concentração mg/mL'], df_cabeçalho['Área'])

summary_of_fit = pd.DataFrame({
    'Medida': [
        'R Quadrado (R²)',
        'Coeficiente de Correlação de Pearson'
    ],
    'Valor': [
        r_squared,
        correlation_coefficient
    ]
})

st.subheader("Tabela 5 - Medidas Descritivas da Qualidade do Ajuste")
st.dataframe(summary_of_fit)

correlation_threshold = 0.9900
if correlation_coefficient > correlation_threshold:
    st.success(f"Correlação ({correlation_coefficient:.4f}) > {correlation_threshold:.4f}: Relação linear adequada.")
else:
    st.warning(f"Correlação ({correlation_coefficient:.4f}) ≤ {correlation_threshold:.4f}: Relação linear pode não ser adequada.")

st.header("4 - Análise Gráfica")

st.subheader("4.1 - Diagrama de Dispersão")

# Gráfico de dispersão
fitted_values = model_sm.fittedvalues
prstd, iv_l_mean, iv_u_mean = wls_prediction_std(model_sm, alpha=0.05)

fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df_cabeçalho, x='Concentração mg/mL', y='Área', label='Dados Originais', ax=ax1)
ax1.plot(df_cabeçalho['Concentração mg/mL'], fitted_values, color='red', label='Linha Ajustada')
ax1.plot(df_cabeçalho['Concentração mg/mL'], iv_l_mean, color='green', linestyle='--', label='IC 95% Inferior')
ax1.plot(df_cabeçalho['Concentração mg/mL'], iv_u_mean, color='purple', linestyle='--', label='IC 95% Superior')
ax1.set_title('Diagrama de Dispersão: Concentração vs Área')
ax1.set_xlabel('Concentração mg/mL')
ax1.set_ylabel('Área')
ax1.grid(True)
ax1.legend()

st.pyplot(fig1)

st.subheader("4.2 - Diagnóstico dos Resíduos")

# Resíduos padronizados vs valores ajustados
standardized_residuals = model_sm.get_influence().resid_studentized_internal
residuals = model_sm.resid

fig2, ((ax2, ax3), (ax4, ax5)) = plt.subplots(2, 2, figsize=(15, 10))

# 1. Resíduos Padronizados vs Valores Ajustados
sns.scatterplot(x=fitted_values, y=standardized_residuals, ax=ax2)
ax2.axhline(y=0, color='r', linestyle='--')
ax2.axhline(y=2, color='blue', linestyle='--')
ax2.axhline(y=-2, color='blue', linestyle='--')
ax2.axhline(y=3, color='red', linestyle='--')
ax2.axhline(y=-3, color='red', linestyle='--')
ax2.set_title('Resíduos Padronizados vs Valores Ajustados')
ax2.set_xlabel('Valores Ajustados')
ax2.set_ylabel('Resíduos Padronizados')
ax2.grid(True)

# 2. QQ-Plot
sm.qqplot(residuals, line='s', ax=ax3)
ax3.set_title('QQ-Plot dos Resíduos')
ax3.grid(True)

# 3. Resíduos vs Valores Ajustados
sns.scatterplot(x=fitted_values, y=residuals, ax=ax4)
ax4.axhline(y=0, color='r', linestyle='--')
ax4.set_title('Resíduos vs Valores Ajustados')
ax4.set_xlabel('Valores Ajustados')
ax4.set_ylabel('Resíduos')
ax4.grid(True)

# 4. Resíduos vs Ordem de Coleta
sns.scatterplot(x=df_cabeçalho['Ordem de Coleta'], y=residuals, ax=ax5)
sns.lineplot(x=df_cabeçalho['Ordem de Coleta'], y=residuals, ax=ax5)
ax5.axhline(y=0, color='r', linestyle='--')
ax5.set_title('Resíduos vs Ordem de Coleta')
ax5.set_xlabel('Ordem de Coleta')
ax5.set_ylabel('Resíduos')
ax5.grid(True)

plt.tight_layout()
st.pyplot(fig2)

st.header("5 - Testes Estatísticos dos Resíduos")

# Teste de Shapiro-Wilk
st.subheader("5.1 - Teste de Normalidade (Shapiro-Wilk)")
shapiro_stat, shapiro_p = stats.shapiro(residuals)

shapiro_results = pd.DataFrame({
    'Teste': ["Shapiro-Wilk"],
    'Estatística': [shapiro_stat],
    'P-valor': [shapiro_p]
})

st.dataframe(shapiro_results)

if shapiro_p > alpha:
    st.success(f"P-valor ({shapiro_p:.4f}) > {alpha}: Não rejeitamos H0. Resíduos seguem distribuição normal.")
else:
    st.warning(f"P-valor ({shapiro_p:.4f}) ≤ {alpha}: Rejeitamos H0. Resíduos podem não seguir distribuição normal.")

# Teste de Cochran para homocedasticidade
st.subheader("5.2 - Teste de Homocedasticidade (Cochran)")
grouped_variances = df_cabeçalho.groupby('Nível')['Área'].var()
max_variance = grouped_variances.max()
sum_variances = grouped_variances.sum()
cochran_c = max_variance / sum_variances

cochran_results = pd.DataFrame({
    'Teste': ["Cochran C"],
    'Estatística': [cochran_c],
    'Observação': ["Consulte tabelas críticas para interpretação formal"]
})

st.dataframe(cochran_results)
st.info(f"Estatística de Cochran C: {cochran_c:.4f}")

st.header("6 - Resumo e Conclusões")

# Equação da reta
intercept = model_sm.params['Intercept']
slope = model_sm.params['Q("Concentração mg/mL")']

st.subheader("Equação da Reta de Regressão:")
st.write(f"**y = {intercept:.4f} + {slope:.4f} × Concentração mg/mL**")

# Resumo das conclusões
st.subheader("Resumo das Conclusões")

conclusions_data = {
    'Teste Realizado': [
        'Teste F da ANOVA (Significância do Modelo)',
        'Teste t do Intercepto',
        'Coeficiente de Correlação de Pearson',
        'Teste de Normalidade dos Resíduos (Shapiro-Wilk)',
        'Teste de Homocedasticidade (Cochran)',
        'Valores Extremos na Resposta'
    ],
    'Conclusão': [
        'Modelo estatisticamente significativo' if p_value_concentration < alpha else 'Modelo não significativo',
        'Intercepto estatisticamente significativo' if intercept_p_value < alpha else 'Intercepto não significativo',
        'Relação linear adequada' if correlation_coefficient > correlation_threshold else 'Relação linear questionável',
        'Resíduos seguem distribuição normal' if shapiro_p > alpha else 'Resíduos podem não ser normais',
        'Consultar valores críticos para conclusão formal',
        'Verificar gráficos de resíduos padronizados (valores > |3| são outliers)'
    ]
}

summary_table = pd.DataFrame(conclusions_data)
st.dataframe(summary_table)

# Informações adicionais
st.sidebar.header("Informações do Modelo")
st.sidebar.write(f"**R²:** {r_squared:.4f}")
st.sidebar.write(f"**Correlação:** {correlation_coefficient:.4f}")
st.sidebar.write(f"**P-valor F:** {p_value_concentration:.4f}")
st.sidebar.write(f"**Intercepto:** {intercept:.4f}")
st.sidebar.write(f"**Inclinação:** {slope:.4f}")

st.sidebar.header("Critérios de Aceitação")
st.sidebar.write("- **Correlação:** > 0.9900")
st.sidebar.write("- **P-valor:** < 0.05")
st.sidebar.write("- **Resíduos:** Normalmente distribuídos")
st.sidebar.write("- **Homocedasticidade:** Variâncias homogêneas")

# Rodapé
st.markdown("---")
st.write("**Nota:** Este aplicativo é para fins educacionais e de demonstração. Para análises formais, consulte um estatístico.")
