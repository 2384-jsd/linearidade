# -*- coding: utf-8 -*-
"""
Aplicativo Streamlit para Análise de Linearidade na Validação Analítica

Para executar: streamlit run linearidade.py
"""

# Verificar se está rodando no Streamlit
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit não está disponível. Execute: pip install streamlit")

if STREAMLIT_AVAILABLE:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.stats as stats
    import seaborn as sns
    import statsmodels.api as sm
    from scipy.stats import pearsonr
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
    try:
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

        st.subheader("Tabela 4 - Medidas Descritivas da Qualidade do Ajuste")
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
        
        try:
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
        except Exception as e:
            # Gráfico simplificado caso haja erro
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df_cabeçalho, x='Concentração mg/mL', y='Área', label='Dados Originais', ax=ax1)
            ax1.plot(df_cabeçalho['Concentração mg/mL'], fitted_values, color='red', label='Linha Ajustada')
            ax1.set_title('Diagrama de Dispersão: Concentração vs Área')
            ax1.set_xlabel('Concentração mg/mL')
            ax1.set_ylabel('Área')
            ax1.grid(True)
            ax1.legend()
            st.pyplot(fig1)

        st.subheader("4.2 - Diagnóstico dos Resíduos")

        # Resíduos
        try:
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
            ax5.axhline(y=0, color='r', linestyle='--')
            ax5.set_title('Resíduos vs Ordem de Coleta')
            ax5.set_xlabel('Ordem de Coleta')
            ax5.set_ylabel('Resíduos')
            ax5.grid(True)

            plt.tight_layout()
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"Erro ao gerar gráficos de resíduos: {e}")

        st.header("5 - Testes Estatísticos dos Resíduos")

        # Teste de Shapiro-Wilk
        try:
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
        except Exception as e:
            st.error(f"Erro no teste de Shapiro-Wilk: {e}")

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
                'Análise Visual dos Resíduos'
            ],
            'Conclusão': [
                'Modelo estatisticamente significativo' if p_value_concentration < alpha else 'Modelo não significativo',
                'Intercepto estatisticamente significativo' if intercept_p_value < alpha else 'Intercepto não significativo',
                'Relação linear adequada' if correlation_coefficient > correlation_threshold else 'Relação linear questionável',
                'Verificar gráficos de resíduos para normalidade e homocedasticidade'
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

    except Exception as e:
        st.error(f"Erro ao executar análise: {e}")
        st.info("Verifique se todas as bibliotecas estão instaladas corretamente.")

    # Critérios de aceitação
    st.sidebar.header("Critérios de Aceitação")
    st.sidebar.write("- **Correlação:** > 0.9900")
    st.sidebar.write("- **P-valor:** < 0.05")
    st.sidebar.write("- **Resíduos:** Normalmente distribuídos")
    st.sidebar.write("- **Homocedasticidade:** Variâncias homogêneas")

    # Rodapé
    st.markdown("---")
    st.write("**Como executar:** `streamlit run linearidade.py`")
    st.write("**Dependências:** Execute `pip install -r requirements.txt`")
    st.write("**Nota:** Este aplicativo é para fins educacionais e de demonstração. Para análises formais, consulte um estatístico.")

else:
    print("Este script foi adaptado para Streamlit.")
    print("Para executar:")
    print("1. pip install -r requirements.txt")
    print("2. streamlit run linearidade.py")

# Título da aplicação
st.title('Linearidade para Validação Analítica')

st.write('''#### Realizar testes estatísticos da linearidade da Validação Analítica de acordo com a RDC 166.

Este é um requisito para os métodos que foram desenvolvidos e estão sendo utilizados para liberação de resultados.
''')

# Sidebar para entrada de dados
st.sidebar.header("Configurações dos Dados")

# Opção para usar dados exemplo ou inserir dados personalizados
use_example_data = st.sidebar.checkbox("Usar dados de exemplo", value=True)

if not use_example_data:
    st.sidebar.info("Funcionalidade de entrada personalizada será implementada em versão futura")

# Tratamento Estatístico - Validação Analítica
st.header("1 - Introdução")

st.write("""
A linearidade de um procedimento analítico é a sua capacidade de obter resultados que sejam diretamente proporcionais à concentração de um analito em uma amostra.
""")

st.header("2 - Coleta de Dados")
st.write("A seguir, apresentam-se os dados coletados:")

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

# Configurações do pandas
pd.options.display.float_format = '{:.4f}'.format
# Criar DataFrame com os dados
df_cabeçalho = pd.DataFrame(dados, columns=['Concentração mg/mL', 'Área', 'Nível', 'Ordem de Coleta'])

# Mostrar tabela de dados
st.subheader("Tabela 1 - Conjunto de dados para o estudo de Linearidade")
st.dataframe(df_cabeçalho)

st.header("3 - Método dos Mínimos Quadrados Ordinários: Estimação")

st.write("""
O Método dos Mínimos Quadrados é uma eficiente estratégia de estimação dos parâmetros da regressão e sua aplicação não é limitada apenas às relações lineares. Nesta seção utilizamos o Método dos Mínimos Quadrados Ordinários.
""")

st.subheader("3.1 - Teste do Coeficiente Angular")
st.write("""
Para avaliar a significância do modelo utilizamos o teste F da ANOVA. Neste caso, testa-se as hipóteses:

H0: coeficiente angular igual a zero;

H1: coeficiente angular diferente de zero.
""")

# Ajustar modelo de regressão linear
model_sm = ols('Área ~ Q("Concentração mg/mL")', data=df_cabeçalho).fit()

# Realizar ANOVA
anova_table = anova_lm(model_sm)

# Preparar tabela da ANOVA para exibição
anova_summary = anova_table.loc[['Q("Concentração mg/mL")', 'Residual'],
                                ['df', 'sum_sq', 'mean_sq', 'F', 'PR(>F)']]
anova_summary.columns = ['Graus de Liberdade', 'Soma dos Quadrados', 'Quadrado Médio', 'Estatística F', 'P-valor']

st.subheader("Tabela 2 - Tabela da ANOVA")
st.dataframe(anova_summary)

# Interpretar o p-valor
alpha = 0.05
p_value_concentration = anova_summary.loc['Q("Concentração mg/mL")', 'P-valor']

st.subheader("Interpretação do Teste F da ANOVA:")
if p_value_concentration < alpha:
    st.success(f"Como o P-valor ({p_value_concentration:.4f}) é menor que {alpha}, rejeitamos a hipótese nula (coeficiente angular igual a zero) ao nível de significância de 5%.")
    st.success("Conclusão: O coeficiente angular é estatisticamente diferente de zero, indicando que a Concentração mg/mL é um preditor significativo da Área.")
else:
    st.warning(f"Como o P-valor ({p_value_concentration:.4f}) não é menor que {alpha}, falhamos em rejeitar a hipótese nula (coeficiente angular igual a zero) ao nível de significância de 5%.")
    st.warning("Conclusão: Não há evidência estatística suficiente para concluir que o coeficiente angular é diferente de zero.")

# Tabela de Coeficientes
st.subheader("Tabela 3 - Coeficientes")
results_summary = model_sm.summary2().tables[1]
coefficients_table = results_summary.loc[['Intercept', 'Q("Concentração mg/mL")'],
                                         ['Coef.', 'Std.Err.', 't', 'P>|t|']]
coefficients_table.columns = ['Estimativa', 'Desvio Padrão', 'Estatística t', 'P-valor']
st.dataframe(coefficients_table)

# Rename columns for clarity
coefficients_table.columns = ['Estimativa', 'Desvio Padrão', 'Estatística t', 'P-valor']

import pandas as pd

# Set pandas options to display floats with one decimal place
pd.options.display.float_format = '{:.4f}'.format

print("Pandas display options for float format have been set to one decimal place.")

print("Tabela de Coeficientes:")
display(coefficients_table)

"""Tabela 4 - Intervalo de confiança para os parâmetros"""

import pandas as pd

# Assuming the 'model_sm' variable from cell 9hv9fsWIlgyu holds the statsmodels OLS model
# If not, you would need to fit the model again:
# model_sm = ols('Área ~ Q("Concentração mg/mL")', data=df_cabeçalho).fit()

# Get the regression results summary
results_summary = model_sm.summary2().tables[1]

# Extract the relevant rows for Intercept and Q("Concentração mg/mL") and columns for confidence intervals
confidence_interval_table = results_summary.loc[['Intercept', 'Q("Concentração mg/mL")'],
                                                ['Coef.', '[0.025', '0.975]']]

# Rename columns for clarity (assuming a 95% confidence interval by default)
confidence_interval_table.columns = ['Estimativa', 'Lower 95% CI', 'Upper 95% CI']

# Set pandas options to display floats with one decimal place
pd.options.display.float_format = '{:.4f}'.format

print("Tabela de Intervalos de Confiança:")
display(confidence_interval_table)

"""##3.2 - Teste do Intercepto (Coeficiente Linear)

Para avaliarmos o intercepto (coeficiente linear) utiliza-se a estatística t de Student. Neste caso, testamos as hipóteses:

H0: intercepto (coeficiente linear) igual a zero;

H1: intercepto diferente de zero.
"""

import pandas as pd

# Assuming the 'model_sm' variable from cell 9hv9fsWIlgyu holds the statsmodels OLS model
# If not, you would need to fit the model again:
# model_sm = ols('Área ~ Q("Concentração mg/mL")', data=df_cabeçalho).fit()

# Get the summary of the fitted model
model_summary = model_sm.summary2().tables[1]

# Extract the p-value for the Intercept
intercept_p_value = model_summary.loc['Intercept', 'P>|t|']

# Define the significance level
alpha = 0.05

# Print the p-value and conclusion
print(f"P-valor para o Intercepto: {intercept_p_value:.4f}")

print("\nInterpretação do Teste t para o Intercepto:")
if intercept_p_value > alpha:
    print(f"Como o P-valor ({intercept_p_value:.4f}) é maior que {alpha}, falhamos em rejeitar a hipótese nula (intercepto igual a zero) ao nível de significância de 5%.")
    print("Conclusão: Não há evidência estatística suficiente para concluir que o intercepto é diferente de zero.")
else:
    print(f"Como o P-valor ({intercept_p_value:.4f}) é menor ou igual a {alpha}, rejeitamos a hipótese nula (intercepto igual a zero) ao nível de significância de 5%.")
    print("Conclusão: O intercepto é estatisticamente diferente de zero.")

"""##3.3 - Impacto do Coeficiente Linear (Intercepto)

Tabela 5 - Medida descritiva da qualidade do ajuste
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# Assuming 'df_cabeçalho' is your pandas DataFrame
# Assuming 'Concentração mg/mL' is the independent variable and 'Área' is the dependent variable

# Fit a linear regression model using sklearn to easily get residuals and R-squared
# We use sklearn here for simplicity in getting R-squared and residuals for standard deviation calculation.
# The statsmodels 'model_sm' fitted earlier can also be used for these values.
X = df_cabeçalho[['Concentração mg/mL']]
y = df_cabeçalho['Área']

model_sklearn = LinearRegression()
model_sklearn.fit(X, y)

# Get the residuals from the sklearn model
residuals = y - model_sklearn.predict(X)

# Calculate the standard deviation of residuals
# Use ddof=model_sklearn.rank_ for the unbiased estimator (n - p)
std_dev_residuals = np.std(residuals, ddof=model_sklearn.rank_)

# Get the degrees of freedom for the residuals
# For simple linear regression, df = n - p - 1, where n is number of observations and p is number of predictors (1)
n = len(df_cabeçalho)
p = X.shape[1]
degrees_freedom = n - p - 1

# Get R-squared from the sklearn model
r_squared = model_sklearn.score(X, y)

# Calculate the Pearson correlation coefficient between 'Concentração mg/mL' and 'Área'
correlation_coefficient, _ = pearsonr(df_cabeçalho['Concentração mg/mL'], df_cabeçalho['Área'])

# Create a DataFrame for the summary table
summary_of_fit_df = pd.DataFrame({
    'Measure': [
        'Desvio Padrão dos Resíduos',
        'Graus de Liberdade',
        'R Quadrado (R²)',
        'Coeficiente de Correlação de Pearson'
    ],
    'Value': [
        std_dev_residuals,
        degrees_freedom,
        r_squared,
        correlation_coefficient
    ]
})

# Set pandas options to display floats with a reasonable number of decimal places
pd.options.display.float_format = '{:.4f}'.format

print("Tabela - Medida Descritiva da Qualidade do Ajuste:")
display(summary_of_fit_df)

# Conclude based on the correlation coefficient
correlation_threshold = 0.9900

print("\nConclusão sobre o Coeficiente de Correlação de Pearson:")
if correlation_coefficient > correlation_threshold:
    print(f"O coeficiente de correlação de Pearson ({correlation_coefficient:.4f}) é maior que {correlation_threshold:.4f}.")
    print("Conclusão: Existe uma relação linear adequada entre Concentração mg/mL e Área.")
# Conclusão sobre o Coeficiente de Correlação de Pearson
correlation_threshold = 0.9900
if correlation_coefficient > correlation_threshold:
    st.success(f"O coeficiente de correlação de Pearson ({correlation_coefficient:.4f}) é maior que {correlation_threshold:.4f}.")
    st.success("Conclusão: Existe uma relação linear adequada entre Concentração mg/mL e Área.")
else:
    st.warning(f"O coeficiente de correlação de Pearson ({correlation_coefficient:.4f}) não é maior que {correlation_threshold:.4f}.")
    st.warning("Conclusão: Não há evidência suficiente para concluir que existe uma relação linear adequada (com base no critério > 0.9900).")

st.header("4 - Análise Gráfica")

st.subheader("4.1 - Diagrama de dispersão")
st.write("""
O diagrama de dispersão é um gráfico que permite a visualização de uma possível associação entre variáveis quantitativas.
""")

# Figura 1 - Diagrama de Dispersão
fitted_values = model_sm.fittedvalues
prstd, iv_l_mean, iv_u_mean = wls_prediction_std(model_sm, alpha=0.05)

fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df_cabeçalho, x='Concentração mg/mL', y='Área', label='Dados Originais', ax=ax1)
ax1.plot(df_cabeçalho['Concentração mg/mL'], fitted_values, color='red', label='Linha Ajustada')
ax1.plot(df_cabeçalho['Concentração mg/mL'], iv_l_mean, color='green', linestyle='--', label='IC 95% Inferior')
ax1.plot(df_cabeçalho['Concentração mg/mL'], iv_u_mean, color='purple', linestyle='--', label='IC 95% Superior')
ax1.set_title('Diagrama de Dispersão: Concentração vs Área com Linha de Tendência e Intervalo de Confiança')
ax1.set_xlabel('Concentração mg/mL')
ax1.set_ylabel('Área')
ax1.grid(True)
ax1.legend()

st.pyplot(fig1)
plt.plot(df_cabeçalho['Concentração mg/mL'], iv_u_mean, color='purple', linestyle='--', label='Upper 95% CI for Mean')


plt.title('Diagrama de Dispersão: Concentração vs Área com Linha de Tendência e Intervalo de Confiança')
plt.xlabel('Concentração mg/mL')
plt.ylabel('Área')
plt.grid(True)
plt.legend()
plt.show()

print("The dashed green and purple lines represent the lower and upper bounds of the 95% confidence interval for the mean response, respectively. This might be what was intended by 'two more lines in the diagonal that are the adjusted values'.")

# Diagrama de dispersão alternativo
fig2, ax2 = plt.subplots(figsize=(8, 6))
df_cabeçalho.plot(kind='scatter', x='Concentração mg/mL', y='Área', s=32, alpha=.8, ax=ax2)
sns.regplot(data=df_cabeçalho, x='Concentração mg/mL', y='Área', scatter=False, color='red', marker='o', ax=ax2)
ax2.spines[['top', 'right']].set_visible(False)
ax2.set_title('Diagrama de Dispersão')
ax2.set_xlabel('Concentração (mg/mL)')
ax2.set_ylabel('Área')
ax2.grid()

st.pyplot(fig2)

st.subheader("4.2 - Diagnóstico dos Resíduos do Modelo")

st.write("""
**I - Gráfico de Resíduos padronizados vs Valores ajustados**: é usado para detectar se os resíduos se distribuem aleatoriamente e para detectar a presença de valores extremos (outliers) nos dados. Geralmente, consideram-se outliers os pontos que excedem o limite de 3 desvios padrão.
""")

# 1. Resíduos Padronizados vs. Valores Ajustados
fitted_values = model_sm.fittedvalues
standardized_residuals = model_sm.get_influence().resid_studentized_internal

fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=fitted_values, y=standardized_residuals, ax=ax3)
ax3.axhline(y=0, color='r', linestyle='--')
ax3.axhline(y=2, color='blue', linestyle='--')
ax3.axhline(y=-2, color='blue', linestyle='--')
ax3.axhline(y=3, color='red', linestyle='--')
ax3.axhline(y=-3, color='red', linestyle='--')
ax3.set_title('Resíduos Padronizados vs. Valores Ajustados')
ax3.set_xlabel('Valores Ajustados')
ax3.set_ylabel('Resíduos Padronizados')
ax3.grid(True)

st.pyplot(fig3)

st.write("""
**II - Gráfico de Resíduos da Normal**: é usado para verificar a pressuposição de que os resíduos são distribuídos normalmente. Em caso de normalidade os resíduos em geral seguem aproximadamente uma linha reta.
""")

# QQ-Plot dos Resíduos
residuals = model_sm.resid
fig4, ax4 = plt.subplots(figsize=(8, 6))
sm.qqplot(residuals, line='s', ax=ax4)
ax4.set_title('QQ-Plot dos Resíduos')
ax4.set_xlabel('Quantis Teóricos da Distribuição Normal')
ax4.set_ylabel('Quantis da Amostra (Resíduos)')
ax4.grid(True)

st.pyplot(fig4)

st.write("""
**III - Gráfico de Resíduos vs Valores ajustados**: é usado para verificar a pressuposição de que os resíduos se distribuem aleatoriamente e que tem variância constante.
""")

# 3. Resíduos vs. Valores Ajustados
fig5, ax5 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=fitted_values, y=residuals, ax=ax5)
ax5.axhline(y=0, color='r', linestyle='--')
ax5.set_title('Resíduos vs. Valores Ajustados')
ax5.set_xlabel('Valores Ajustados')
ax5.set_ylabel('Resíduos')
ax5.grid(True)

st.pyplot(fig5)

st.write("""
**IV - Gráfico de Resíduos vs Ordem de coleta**: este gráfico mostra os resíduos na ordem em que foram coletados e é usado para verificar a pressuposição de independência. Em geral, para que se cumpra tal requisito, os dados devem se dispor aleatoriamente em torno da linha central.
""")

# 4. Resíduos vs. Ordem de Coleta
fig6, ax6 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=df_cabeçalho['Ordem de Coleta'], y=residuals, ax=ax6)
sns.lineplot(x=df_cabeçalho['Ordem de Coleta'], y=residuals, ax=ax6)
ax6.axhline(y=0, color='r', linestyle='--')
ax6.set_title('Resíduos vs. Ordem de Coleta')
ax6.set_xlabel('Ordem de Coleta')
ax6.set_ylabel('Resíduos')
ax6.grid(True)

st.pyplot(fig6)

st.header("5 - Avaliação dos Resíduos")

st.subheader("5.1 - Avaliação da Normalidade: teste de Shapiro-Wilk")

st.write("""
Para avaliar estatisticamente a normalidade dos resíduos realiza-se o seguinte teste de hipóteses:

H0: a distribuição dos resíduos é Normal;

H1: a distribuição dos resíduos não é Normal.
""")

import pandas as pd
import scipy.stats as stats

# Assuming 'residuals' variable holds the residuals from the fitted model
# If not, you would need to calculate or get the residuals first.
# For example, if you have a statsmodels model object named 'model_sm':
# residuals = model_sm.resid

# Perform the Shapiro-Wilk test for normality
shapiro_test_statistic, shapiro_p_value = stats.shapiro(residuals)

# Create a DataFrame to display the results
shapiro_results_df = pd.DataFrame({
    'Test Performed': ["Shapiro-Wilk Test for Normality of Residuals"],
    'Test Statistic': [shapiro_test_statistic],
    'P-value': [shapiro_p_value]
})

# Set pandas options to display floats with a reasonable number of decimal places
pd.options.display.float_format = '{:.4f}'.format

print("Shapiro-Wilk Test Results for Residuals:")
display(shapiro_results_df)

# Interpret the p-value
alpha = 0.05

print("\nInterpretation:")
print("H0: A distribuição dos resíduos é Normal.")
print("H1: A distribuição dos resíduos não é Normal.")

if shapiro_p_value > alpha:
    print(f"Como o P-valor ({shapiro_p_value:.4f}) é maior que {alpha}, falhamos em rejeitar a hipótese nula (a distribuição dos resíduos é Normal) ao nível de significância de 5%.")
    print("Conclusão: Não há evidência estatística suficiente para concluir que a distribuição dos resíduos não é Normal. Assume-se normalidade dos resíduos.")
else:
    print(f"Como o P-valor ({shapiro_p_value:.4f}) é menor ou igual a {alpha}, rejeitamos a hipótese nula (a distribuição dos resíduos é Normal) ao nível de significância de 5%.")
    print("Conclusão: Há evidência estatística para concluir que a distribuição dos resíduos não é Normal.")

"""
## 5.2 - Avaliação da Homocedasticidade:  Teste de Cochran

Para avaliarmos a homocedasticidade da variância realiza-se o seguinte teste de hipóteses:

H0: Variâncias dos níveis são iguais;
H1: Pelo menos uma variância diferente.

A seguir, apresentamos o teste de Teste de Cochran."""

import numpy as np
import pandas as pd
from scipy.stats import f

# Assuming 'df_cabeçalho' is your pandas DataFrame
# Assuming 'Nível' is the grouping variable and 'Área' is the variable to test for homoscedasticity

# Group data by 'Nível' and calculate variances of 'Área'
grouped_variances = df_cabeçalho.groupby('Nível')['Área'].var().reset_index()
grouped_variances.rename(columns={'Área': 'Variance'}, inplace=True)

# Get the maximum variance
max_variance = grouped_variances['Variance'].max()

# Get the sum of variances
sum_of_variances = grouped_variances['Variance'].sum()

# Calculate Cochran's C statistic
cochran_c_statistic = max_variance / sum_of_variances

# Number of groups (k) and replicates per group (n)
k = len(grouped_variances) # Number of levels
n = 3 # Number of replicates per level, as specified by the user

# Calculate p-value for Cochran's C test (approximate method using F distribution)
# This is an approximation and might not be as accurate as using exact tables or dedicated functions.
# For a more precise p-value, dedicated statistical libraries or tables for Cochran's C are recommended.
# Using an approximation based on the maximum variance and average variance:
# F-statistic approximation: ((n-1)*C) / (1-C) where C is Cochran's C
if cochran_c_statistic < 1: # Ensure C is less than 1 for the formula
    f_approx = ((n - 1) * cochran_c_statistic) / (1 - cochran_c_statistic)
    # Degrees of freedom for the approximation: df1 = k-1, df2 = k*(n-1)
    df1 = k - 1
    df2 = k * (n - 1)
    # Calculate the approximate p-value (upper tail)
    p_value_cochran = 1 - f.cdf(f_approx, df1, df2)
else:
    p_value_cochran = 0.0 # If C >= 1, it's highly likely to reject H0

# Create a DataFrame for the results
cochran_results_df = pd.DataFrame({
    'Test Performed': ["Cochran's C Test for Homoscedasticity"],
    'Test Statistic': [cochran_c_statistic],
    'Number of Replicates (n)': [n],
    'Approximate P-value': [p_value_cochran]
})

# Add the conclusion
alpha = 0.05
print("\nHypotheses for Cochran's C Test:")
print("H0: Variances of the levels are equal (Homoscedasticity)")
print("H1: At least one variance is different (Heteroscedasticity)")


print("\nCochran's C Test Results:")
# Set pandas options to display floats with a reasonable number of decimal places for the statistic
pd.options.display.float_format = '{:.4f}'.format
display(cochran_results_df)


print("\nInterpretation:")
if p_value_cochran < alpha:
    print(f"As the approximate P-valor ({p_value_cochran:.4f}) é menor que {alpha}, rejeitamos a hipótese nula (Variâncias dos níveis são iguais) ao nível de significância de 5%.")
    print("Conclusão: Há evidência estatística para concluir que pelo menos uma variância é diferente. Logo, temos um modelo heterocedástico.")
else:
    print(f"As the approximate P-valor ({p_value_cochran:.4f}) não é menor que {alpha}, falhamos em rejeitar a hipótese nula (Variâncias dos níveis são iguais) ao nível de significância de 5%.")
    print("Conclusão: Não há evidência estatística suficiente para concluir que as variâncias são diferentes. Assume-se homoscedasticidade.")

"""## 6 - Valores extremos na resposta

Nesta seção, avalia-se os valores extremos (outliers) na resposta. Para isso, analisa-se os resíduos padronizados.

## 6.1 - Resíduos

Como critério, são considerados valores extremos nas resposta, as observações com resíduos padronizados maiores que 3.

Tabela 8 - Resumo da Análise dos Resíduos
"""

import pandas as pd

# Get the residuals from the fitted model (using the statsmodels OLS model)
# Assuming the 'model_sm' variable from cell TcUvkEEmfOIv holds the statsmodels OLS model
residuals = model_sm.resid

# Get standardized residuals (internally studentized)
standardized_residuals = model_sm.get_influence().resid_studentized_internal

# Get studentized residuals (externally studentized)
studentized_residuals = model_sm.get_influence().resid_studentized

# Create a DataFrame to display the residuals
residuals_df = pd.DataFrame({
    'Residuals': residuals,
    'Standardized Residuals': standardized_residuals,
    'Studentized Residuals': studentized_residuals
})

# Display the DataFrame
display(residuals_df)

# 3. Resíduos vs. Valores Ajustados
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Get the fitted values from the statsmodels OLS model
# Assuming the 'model_sm' variable from cell TcUvkEEmfOIv holds the statsmodels OLS model
fitted_values = model_sm.fittedvalues

# Get the residuals from the fitted model
# Assuming the 'model_sm' variable from cell TcUvkEEmfOIv holds the statsmodels OLS model
residuals = model_sm.resid

plt.figure(figsize=(6, 4))
sns.scatterplot(x=fitted_values, y=residuals)
plt.axhline(y=0, color='r', linestyle='--') # Add line at 0
plt.title('Resíduos vs. Valores Ajustados')
plt.xlabel('Valores Ajustados')
plt.ylabel('Resíduos')
plt.grid(True)
plt.show()

"""## 7 - Teste de Independência

Neste item testamos a independência das observações através do seguinte teste de hipóteses:

H0: Observações são independentes;

H1: Observações não são independentes.
"""

import pandas as pd
import statsmodels.api as sm

# Assuming the 'model_sm' variable from cell TcUvkEEmfOIv holds the statsmodels OLS model
# If not, you would need to fit the model again:
# model_sm = ols('Área ~ Q("Concentração mg/mL")', data=df_cabeçalho).fit()

# Get the Durbin-Watson statistic from the model summary
# This accesses the statistic from the summary table as demonstrated in the previous successful cell.
try:
    durbin_watson_statistic = model_sm.get_robustcov_results(cov_type='HC1').summary().tables[0].data[6][1]
    # Attempt to convert to float for display and comparison
    durbin_watson_statistic_float = float(durbin_watson_statistic)
except (AttributeError, IndexError, KeyError, ValueError, TypeError):
    durbin_watson_statistic = "Could not retrieve statistic"
    durbin_watson_statistic_float = None


# Create a DataFrame for display. Note: No standard p-value is directly available.
durbin_watson_df = pd.DataFrame({
    'Test Performed': ["Durbin-Watson Test for Independence of Observations"],
    'Test Statistic': [durbin_watson_statistic],
    'Note on P-value': ["A p-valor padrão não é fornecido diretamente na saída do summary2() para este teste. A interpretação formal requer valores críticos."]
})

print("Durbin-Watson Test Results:")
display(durbin_watson_df)

print("\nInterpretation:")
print("A estatística de Durbin-Watson varia de 0 a 4.")
print("- Um valor próximo a 2 sugere que não há autocorrelação nos resíduos (observações são independentes).")
print("- Um valor significativamente menor que 2 sugere autocorrelação positiva.")
print("- Um valor significativamente maior que 2 sugere autocorrelação negativa.")

if durbin_watson_statistic_float is not None:
    print(f"\nEstatística de Durbin-Watson obtida: {durbin_watson_statistic_float:.4f}")

    # Provide the standard interpretation based on the statistic's proximity to 2.
    # This is an approximate guideline, not a formal hypothesis test conclusion with a p-value.
    if durbin_watson_statistic_float > 1.5 and durbin_watson_statistic_float < 2.5:
         print("Conclusão (Baseada na Estatística): A estatística de Durbin-Watson está próxima de 2, sugerindo que não há evidência forte de autocorrelação nos resíduos.")
    else:
         print("Conclusão (Baseada na Estatística): A estatística de Durbin-Watson sugere possível autocorrelação nos resíduos. Recomenda-se consultar tabelas de valores críticos para uma conclusão formal com base no nível de significância de 5%.")
else:
    print("\nNão foi possível obter a estatística de Durbin-Watson para interpretação.")

print("\nPara uma conclusão formal ao nível de significância de 5%, compare a estatística de Durbin-Watson com os valores críticos de tabelas apropriadas para o tamanho da sua amostra e número de preditores.")

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Assuming the 'model_sm' variable from cell 9hv9fsWIlgyu holds the statsmodels OLS model
# Get the fitted values from the statsmodels OLS model
fitted_values = model_sm.fittedvalues

# Calculate standardized residuals
standardized_residuals = model_sm.get_influence().resid_studentized_internal

# 1. Standardized Residuals vs. Fitted Values Plot
plt.figure(figsize=(8, 4))
sns.scatterplot(x=fitted_values, y=standardized_residuals)
plt.axhline(y=0, color='r', linestyle='--') # Add line at 0
plt.axhline(y=2, color='blue', linestyle='--') # Add line at +2
plt.axhline(y=-2, color='blue', linestyle='--') # Add line at -2
plt.axhline(y=3, color='red', linestyle='--') # Add line at +3
plt.axhline(y=-3, color='red', linestyle='--') # Add line at -3
plt.title('Resíduos Padronizados vs. Valores Ajustados')
plt.xlabel('Valores Ajustados')
plt.ylabel('Resíduos Padronizados')
plt.grid(True)
plt.show()

# 4. Resíduos vs. Ordem de Coleta
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df_cabeçalho['Ordem de Coleta'], y=residuals)
sns.lineplot(x=df_cabeçalho['Ordem de Coleta'], y=residuals)
plt.axhline(y=0, color='r', linestyle='--') # Add line at 0
plt.title('Resíduos vs. Ordem de Coleta')
plt.xlabel('Ordem de Coleta')
plt.ylabel('Resíduos')
plt.grid(True)
plt.show()

"""## 8 - Resumo do estudo de linearidade

Para finalizarmos, apresentamos uma tabela com o resumo do estudo. O nível de significância utilizado para a conclusão que envolve teste estatístico é de 5%.
"""

# Get the estimated coefficients from the fitted model
# Assuming 'model_sm' holds the fitted statsmodels OLS model
intercept = model_sm.params['Intercept']
slope = model_sm.params['Q("Concentração mg/mL")']

# Display the equation of the line
print(f"Equação da Reta de Regressão:")
print(f"y = {intercept:.4f} + {slope:.4f} * Concentração mg/mL")

"""Tabela 10 - Resumo das conclusões do estudo de Linearidade (método dos mínimos quadrados ordinários)"""

alpha = 0.05

# Ensure necessary variables from previous cells are available
# anova_table from cell 9hv9fsWIlgyu
# coefficients_table from cell 75ybDfutrv3V or _h_gtRoMA09n
# shapiro_results_df from cell 7P6wvsYwcI47
# cochran_results_df from cell PzzNl7zmzSFM
# standardized_residuals from cell uuPd89kOgWuG or zSTSWc9zJU15
# durbin_watson_statistic_float from cell af7_ADtjXjaB

# 1. ANOVA F-test conclusion
if 'anova_table' in locals() and not anova_table.empty:
    if anova_table.loc['Q("Concentração mg/mL")', 'PR(>F)'] < alpha:
        anova_conclusion = "Model is statistically significant"
    else:
        anova_conclusion = "Model is not statistically significant"
else:
    anova_conclusion = "ANOVA results not available"

# 2. Intercept t-test conclusion
if 'coefficients_table' in locals() and not coefficients_table.empty:
    if coefficients_table.loc['Intercept', 'P-valor'] < alpha:
        intercept_conclusion = "Intercept is statistically significant"
    else:
        intercept_conclusion = "Intercept is not statistically significant"
else:
    intercept_conclusion = "Intercept t-test results not available"

# 3. Concentration t-test conclusion
if 'coefficients_table' in locals() and not coefficients_table.empty:
    if coefficients_table.loc['Q("Concentração mg/mL")', 'P-valor'] < alpha:
        concentration_conclusion = "Concentration coefficient is statistically significant"
    else:
        concentration_conclusion = "Concentration coefficient is not statistically significant"
else:
    concentration_conclusion = "Concentration t-test results not available"

# 4. Normality test conclusion for residuals
# Prioritize Shapiro-Wilk test results if available
if 'shapiro_results_df' in locals() and not shapiro_results_df.empty:
     if shapiro_results_df.loc[0, 'P-value'] > alpha:
          normality_conclusion = f"Residuals appear normally distributed (Shapiro-Wilk P-value = {shapiro_results_df.loc[0, 'P-value']:.4f} > 0.05)."
     else:
          normality_conclusion = f"Residuals do not appear normally distributed (Shapiro-Wilk P-value = {shapiro_results_df.loc[0, 'P-value']:.4f} <= 0.05)."
# If Anderson-Darling test results are available, use them as a fallback
elif 'ad_statistic' in locals() and 'ad_critical_value_5' in locals():
    if ad_statistic > ad_critical_value_5:
        normality_conclusion = "Residuals do not appear normally distributed (at 5% significance level based on Anderson-Darling test)"
    else:
        normality_conclusion = "Residuals appear normally distributed (at 5% significance level based on Anderson-Darling test)"
else:
    normality_conclusion = "Normality test results not available"


# 5. Cochran's C test conclusion for homoscedasticity
# Use the approximate p-value from cochran_results_df if available
if 'cochran_results_df' in locals() and not cochran_results_df.empty:
    if cochran_results_df.loc[0, 'Approximate P-value'] < alpha:
        homoscedasticity_conclusion = f"Evidence of heteroscedasticity (Approx. P-value = {cochran_results_df.loc[0, 'Approximate P-value']:.4f} < 0.05)"
    else:
        homoscedasticity_conclusion = f"No strong evidence of heteroscedasticity (Approx. P-value = {cochran_results_df.loc[0, 'Approximate P-value']:.4f} >= 0.05)"
else:
     homoscedasticity_conclusion = "Homoscedasticity test results not available"


# 6. Extreme Values in Response (based on Standardized Residuals)
if 'standardized_residuals' in locals():
    if (standardized_residuals > 3).any() or (standardized_residuals < -3).any():
        extreme_values_conclusion = "Evidence of extreme values (outliers) in the response (> |3| standardized residuals)."
    else:
        extreme_values_conclusion = "No strong evidence of extreme values (outliers) in the response (all standardized residuals <= |3|)."
else:
     extreme_values_conclusion = "Standardized residuals not available for extreme value analysis."

# 7. Independence of Observations (Durbin-Watson)
if 'durbin_watson_statistic_float' in locals() and durbin_watson_statistic_float is not None:
    if durbin_watson_statistic_float > 1.5 and durbin_watson_statistic_float < 2.5:
         independence_conclusion = f"Durbin-Watson statistic ({durbin_watson_statistic_float:.4f}) is near 2, suggesting no strong autocorrelation."
    else:
         independence_conclusion = f"Durbin-Watson statistic ({durbin_watson_statistic_float:.4f}) suggests possible autocorrelation. Requires critical values for formal conclusion."
else:
     independence_conclusion = "Durbin-Watson test results not available."


# Create a list of test names (in Portuguese)
test_names = [
    "Teste F da ANOVA (Significância do Modelo)",
    "Teste t do Intercepto",
    "Teste t do Coeficiente de Concentração", # Added specific name for clarity
    "Teste de Normalidade dos Resíduos",
    "Teste de Homocedasticidade (Teste de Cochran)",
    "Valores Extremos na Resposta",
    "Teste de Independência das Observações (Durbin-Watson)"
]

# Create a list of corresponding conclusions
conclusions = [
    anova_conclusion,
    intercept_conclusion,
    concentration_conclusion, # Use the concentration_conclusion here
    normality_conclusion,
    homoscedasticity_conclusion,
    extreme_values_conclusion,
    independence_conclusion
]

# Create the DataFrame
summary_table = pd.DataFrame({
    "Teste Realizado": test_names,
    "Conclusão": conclusions
})

# Set pandas options to display the full content of the conclusion column
pd.options.display.max_colwidth = None

print("Tabela Resumo das Conclusões do Estudo de Linearidade:")
display(summary_table)

import pandas as pd

# Define the significance level
alpha = 0.05

# Recalculate conclusions for each test
# Ensure necessary variables from previous cells are available (e.g., anova_table, coefficients_table, ad_statistic, ad_critical_value_5, cochran_c_statistic)

# 1. ANOVA F-test conclusion
# Assuming anova_table is available from a previous executed cell (e.g., 9hv9fsWIlgyu)
if 'anova_table' in locals() and not anova_table.empty:
    if anova_table.loc['Q("Concentração mg/mL")', 'PR(>F)'] < alpha:
        anova_conclusion = "Model is statistically significant"
    else:
        anova_conclusion = "Model is not statistically significant"
else:
    anova_conclusion = "ANOVA results not available"

# 2. Intercept t-test conclusion
# Assuming coefficients_table is available from a previous executed cell (e.g., 75ybDfutrv3V or _h_gtRoMA09n)
if 'coefficients_table' in locals() and not coefficients_table.empty:
    if coefficients_table.loc['Intercept', 'P-valor'] < alpha:
        intercept_conclusion = "Intercept is statistically significant"
    else:
        intercept_conclusion = "Intercept is not statistically significant"
else:
    intercept_conclusion = "Intercept t-test results not available"

# 3. Concentration t-test conclusion
# Assuming coefficients_table is available
if 'coefficients_table' in locals() and not coefficients_table.empty:
    if coefficients_table.loc['Q("Concentração mg/mL")', 'P-valor'] < alpha:
        concentration_conclusion = "Concentration coefficient is statistically significant"
    else:
        concentration_conclusion = "Concentration coefficient is not statistically significant"
else:
    concentration_conclusion = "Concentration t-test results not available"


# 4. Anderson-Darling test conclusion for residuals
# Assuming ad_statistic and ad_critical_value_5 are available from a previous executed cell (e.g., 14220d2c)
# Note: Need to ensure cell 14220d2c or equivalent defining these variables is executed.
# If ad_statistic is defined:
if 'ad_statistic' in locals() and 'ad_critical_value_5' in locals():
    if ad_statistic > ad_critical_value_5:
        ad_conclusion = "Residuals do not appear normally distributed (at 5% significance level)"
    else:
        ad_conclusion = "Residuals appear normally distributed (at 5% significance level)"
elif 'shapiro_results_df' in locals() and not shapiro_results_df.empty:
     # Fallback to Shapiro-Wilk if Anderson-Darling results not available
     if shapiro_results_df.loc[0, 'P-value'] > alpha:
          ad_conclusion = f"Residuals appear normally distributed (Shapiro-Wilk P-value = {shapiro_results_df.loc[0, 'P-value']:.4f} > 0.05)."
     else:
          ad_conclusion = f"Residuals do not appear normally distributed (Shapiro-Wilk P-value = {shapiro_results_df.loc[0, 'P-value']:.4f} <= 0.05)."
else:
    ad_conclusion = "Normality test results not available"


# 5. Cochran's C test conclusion for homoscedasticity
# Assuming cochran_c_statistic and approx_cochran_critical_value_05 are available
# Let's use the approximate p-value from cochran_results_df (cell pdBveDINMhR9) as it was executed
if 'cochran_results_df' in locals() and not cochran_results_df.empty:
    if cochran_results_df.loc[0, 'Approximate P-value'] < alpha:
        cochran_conclusion = f"Evidence of heteroscedasticity (Approx. P-value = {cochran_results_df.loc[0, 'Approximate P-value']:.4f} < 0.05)"
    else:
        cochran_conclusion = f"No strong evidence of heteroscedasticity (Approx. P-value = {cochran_results_df.loc[0, 'Approximate P-value']:.4f} >= 0.05)"
else:
     cochran_conclusion = "Homoscedasticity test results not available"


# 6. Extreme Values in Response (based on Standardized Residuals)
# Assuming standardized_residuals is available from cell uuPd89kOgWuG or zSTSWc9zJU15
if 'standardized_residuals' in locals():
    if (standardized_residuals > 3).any() or (standardized_residuals < -3).any():
        extreme_values_conclusion = "Evidence of extreme values (outliers) in the response (> |3| standardized residuals)."
    else:
        extreme_values_conclusion = "No strong evidence of extreme values (outliers) in the response (all standardized residuals <= |3|)."
else:
     extreme_values_conclusion = "Standardized residuals not available for extreme value analysis."

# 7. Independence of Observations (Durbin-Watson)
# Assuming durbin_watson_statistic_float is available from cell af7_ADtjXjaB
if 'durbin_watson_statistic_float' in locals() and durbin_watson_statistic_float is not None:
    if durbin_watson_statistic_float > 1.5 and durbin_watson_statistic_float < 2.5:
         independence_conclusion = f"Durbin-Watson statistic ({durbin_watson_statistic_float:.4f}) is near 2, suggesting no strong autocorrelation."
    else:
         independence_conclusion = f"Durbin-Watson statistic ({durbin_watson_statistic_float:.4f}) suggests possible autocorrelation. Requires critical values for formal conclusion."
else:
     independence_conclusion = "Durbin-Watson test results not available."


# Create a list of test names (in Portuguese)
test_names = [
    "Teste F da ANOVA (Significância do Modelo)",
    "Teste t do Intercepto",
    "Coeficiente de Correlação de Pearson",
    "Teste de Normalidade dos Resíduos",
    "Teste de Homocedasticidade (Teste de Cochran)",
    "Valores Extremos na Resposta",
    "Teste de Independência das Observações (Durbin-Watson)"
]

# Create a list of corresponding conclusions
conclusions = [
    anova_conclusion,
    intercept_conclusion,
    concentration_conclusion,
    ad_conclusion,
    cochran_conclusion,
    extreme_values_conclusion,
    independence_conclusion
]

# Create the DataFrame
summary_table = pd.DataFrame({
    "Teste Realizado": test_names,
    "Conclusão": conclusions
})

# Set pandas options to display the full content of the conclusion column
pd.options.display.max_colwidth = None

print("Tabela Resumo das Conclusões do Estudo de Linearidade:")
display(summary_table)

"""## Display summary table

### Subtask:
Display the created summary table.

**Reasoning**:
Display the created summary table DataFrame.
"""

display(summary_table)

import matplotlib
import numpy
import pandas
import scipy
import seaborn

print(f"NumPy version: {numpy.__version__}")
print(f"Pandas version: {pandas.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Seaborn version: {seaborn.__version__}")
print(f"SciPy version: {scipy.__version__}")