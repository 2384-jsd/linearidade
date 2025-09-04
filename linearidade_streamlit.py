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

# Constantes
DADOS_EXEMPLO = [
    [0.157043,4660341,1,14], [0.160399,4754809,1,12], [0.160879,4681500,1,2],
    [0.176673,5384404,2,7], [0.180449,5428730,2,4], [0.180989,5263780,2,15],
    [0.196304,5843775,3,10], [0.200499,5993295,3,1], [0.201099,5841594,3,11],
    [0.215934,6555505,4,8], [0.220549,6607388,4,6], [0.221209,6294577,4,3],
    [0.235564,7109287,5,9], [0.240599,7181692,5,5], [0.241318,7095422,5,13],
]

COLUNAS_DF = ['Concentração mg/mL', 'Área', 'Nível', 'Ordem de Coleta']
NIVEIS_FIXOS = [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5]
ALPHA = 0.05
CORRELATION_THRESHOLD = 0.9900

def criar_interface_entrada():
    """Criar interface para entrada de dados"""
    st.sidebar.header("Configurações dos Dados")
    use_example_data = st.sidebar.checkbox("Usar dados de exemplo", value=True)
    
    # Opção para minimizar/expandir entrada manual
    if not use_example_data:
        st.sidebar.subheader("Entrada Manual de Dados")
        expand_manual_entry = st.sidebar.checkbox("Expandir formulário de entrada", value=True)
        
        if expand_manual_entry:
            return criar_formulario_manual()
        else:
            st.sidebar.info("Marque 'Expandir formulário de entrada' para inserir dados manualmente")
            return DADOS_EXEMPLO
    
    return DADOS_EXEMPLO

def criar_formulario_manual():
    """Criar formulário para entrada manual de dados"""
    st.sidebar.info("Preencha os dados manualmente. Os valores da coluna 'Nível' são fixos conforme o delineamento experimental.")
    
    with st.form("dados_manuais"):
        st.subheader("Entrada Manual de Dados de Linearidade")
        st.write("**Estrutura fixa:** 15 medições organizadas em 5 níveis de concentração (3 replicatas por nível)")
        
        # Criar listas para armazenar os dados
        concentracoes = []
        areas = []
        ordens_coleta = []
        
        # Criar campos para cada linha
        cols = st.columns(3)
        cols[0].write("**Concentração (mg/mL)**")
        cols[1].write("**Área**")
        cols[2].write("**Ordem de Coleta**")
        
        for i in range(15):
            col1, col2, col3, col4 = st.columns([2, 2, 1, 2])
            
            with col1:
                conc = st.number_input(
                    f"Linha {i+1}",
                    min_value=0.0,
                    value=0.0,
                    step=0.001,
                    format="%.6f",
                    key=f"conc_{i}"
                )
                concentracoes.append(conc)
            
            with col2:
                area = st.number_input(
                    f"Área {i+1}",
                    min_value=0,
                    value=0,
                    step=1,
                    key=f"area_{i}"
                )
                areas.append(area)
            
            with col3:
                st.write(f"**Nível {NIVEIS_FIXOS[i]}**")
            
            with col4:
                ordem = st.number_input(
                    f"Ordem {i+1}",
                    min_value=1,
                    max_value=15,
                    value=i+1,
                    key=f"ordem_{i}"
                )
                ordens_coleta.append(ordem)
        
        submitted = st.form_submit_button("Processar Dados Manuais")
        
        if submitted:
            # Validar se todos os campos foram preenchidos
            if all(c > 0 for c in concentracoes) and all(a > 0 for a in areas):
                # Criar lista de dados no formato esperado
                dados = []
                for i in range(15):
                    dados.append([concentracoes[i], areas[i], NIVEIS_FIXOS[i], ordens_coleta[i]])
                st.success("Dados carregados com sucesso!")
                return dados
            else:
                st.error("Por favor, preencha todos os campos com valores válidos (maiores que zero).")
                return DADOS_EXEMPLO
    
    return DADOS_EXEMPLO

def criar_modelo_regressao(df):
    """Criar e ajustar modelo de regressão linear"""
    return ols('Área ~ Q("Concentração mg/mL")', data=df).fit()

def executar_anova(model):
    """Executar análise ANOVA"""
    anova_table = anova_lm(model)
    anova_summary = anova_table.loc[['Q("Concentração mg/mL")', 'Residual'],
                                    ['df', 'sum_sq', 'mean_sq', 'F', 'PR(>F)']]
    anova_summary.columns = ['Graus de Liberdade', 'Soma dos Quadrados', 'Quadrado Médio', 'Estatística F', 'P-valor']
    return anova_summary

def interpretar_anova(anova_summary):
    """Interpretar resultados da ANOVA"""
    p_value_concentration = anova_summary.loc['Q("Concentração mg/mL")', 'P-valor']
    
    st.subheader("Interpretação do Teste F da ANOVA:")
    if p_value_concentration < ALPHA:
        st.success(f"P-valor ({p_value_concentration:.4f}) < {ALPHA}: Rejeitamos H0. O coeficiente angular é estatisticamente significativo.")
    else:
        st.warning(f"P-valor ({p_value_concentration:.4f}) ≥ {ALPHA}: Não rejeitamos H0. Coeficiente angular não é estatisticamente significativo.")
    
    return p_value_concentration

def analisar_coeficientes(model):
    """Analisar coeficientes do modelo"""
    results_summary = model.summary2().tables[1]
    coefficients_table = results_summary.loc[['Intercept', 'Q("Concentração mg/mL")'],
                                             ['Coef.', 'Std.Err.', 't', 'P>|t|']]
    coefficients_table.columns = ['Estimativa', 'Desvio Padrão', 'Estatística t', 'P-valor']
    
    confidence_interval_table = results_summary.loc[['Intercept', 'Q("Concentração mg/mL")'],
                                                    ['Coef.', '[0.025', '0.975]']]
    confidence_interval_table.columns = ['Estimativa', 'IC 95% Inferior', 'IC 95% Superior']
    
    return coefficients_table, confidence_interval_table

def testar_intercepto(coefficients_table):
    """Testar significância do intercepto"""
    intercept_p_value = coefficients_table.loc['Intercept', 'P-valor']
    st.write(f"**P-valor para o Intercepto:** {intercept_p_value:.4f}")

    if intercept_p_value > ALPHA:
        st.info(f"P-valor ({intercept_p_value:.4f}) > {ALPHA}: Não rejeitamos H0. O intercepto não é estatisticamente diferente de zero.")
    else:
        st.success(f"P-valor ({intercept_p_value:.4f}) ≤ {ALPHA}: Rejeitamos H0. O intercepto é estatisticamente diferente de zero.")
    
    return intercept_p_value

def calcular_qualidade_ajuste(df):
    """Calcular métricas de qualidade do ajuste"""
    X = df[['Concentração mg/mL']]
    y = df['Área']
    model_sklearn = LinearRegression()
    model_sklearn.fit(X, y)

    r_squared = model_sklearn.score(X, y)
    correlation_coefficient, _ = pearsonr(df['Concentração mg/mL'], df['Área'])

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
    
    return summary_of_fit, correlation_coefficient

def interpretar_correlacao(correlation_coefficient):
    """Interpretar coeficiente de correlação"""
    if correlation_coefficient > CORRELATION_THRESHOLD:
        st.success(f"Correlação ({correlation_coefficient:.4f}) > {CORRELATION_THRESHOLD:.4f}: Relação linear adequada.")
    else:
        st.warning(f"Correlação ({correlation_coefficient:.4f}) ≤ {CORRELATION_THRESHOLD:.4f}: Relação linear pode não ser adequada.")

def criar_grafico_dispersao(df, model):
    """Criar gráfico de dispersão com linha ajustada"""
    fitted_values = model.fittedvalues
    prstd, iv_l_mean, iv_u_mean = wls_prediction_std(model, alpha=0.05)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='Concentração mg/mL', y='Área', label='Dados Originais', ax=ax)
    ax.plot(df['Concentração mg/mL'], fitted_values, color='red', label='Linha Ajustada')
    ax.plot(df['Concentração mg/mL'], iv_l_mean, color='green', linestyle='--', label='IC 95% Inferior')
    ax.plot(df['Concentração mg/mL'], iv_u_mean, color='purple', linestyle='--', label='IC 95% Superior')
    ax.set_title('Diagrama de Dispersão: Concentração vs Área')
    ax.set_xlabel('Concentração mg/mL')
    ax.set_ylabel('Área')
    ax.grid(True)
    ax.legend()
    
    return fig

def criar_diagnosticos_residuos(model, df):
    """Criar gráficos de diagnóstico dos resíduos"""
    fitted_values = model.fittedvalues
    standardized_residuals = model.get_influence().resid_studentized_internal
    residuals = model.resid

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Resíduos Padronizados vs Valores Ajustados
    sns.scatterplot(x=fitted_values, y=standardized_residuals, ax=ax1)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.axhline(y=2, color='blue', linestyle='--')
    ax1.axhline(y=-2, color='blue', linestyle='--')
    ax1.axhline(y=3, color='red', linestyle='--')
    ax1.axhline(y=-3, color='red', linestyle='--')
    ax1.set_title('Resíduos Padronizados vs Valores Ajustados')
    ax1.set_xlabel('Valores Ajustados')
    ax1.set_ylabel('Resíduos Padronizados')
    ax1.grid(True)

    # 2. QQ-Plot
    sm.qqplot(residuals, line='s', ax=ax2)
    ax2.set_title('QQ-Plot dos Resíduos')
    ax2.grid(True)

    # 3. Resíduos vs Valores Ajustados
    sns.scatterplot(x=fitted_values, y=residuals, ax=ax3)
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_title('Resíduos vs Valores Ajustados')
    ax3.set_xlabel('Valores Ajustados')
    ax3.set_ylabel('Resíduos')
    ax3.grid(True)

    # 4. Resíduos vs Ordem de Coleta
    sns.scatterplot(x=df['Ordem de Coleta'], y=residuals, ax=ax4)
    sns.lineplot(x=df['Ordem de Coleta'], y=residuals, ax=ax4)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_title('Resíduos vs Ordem de Coleta')
    ax4.set_xlabel('Ordem de Coleta')
    ax4.set_ylabel('Resíduos')
    ax4.grid(True)

    plt.tight_layout()
    return fig

def executar_testes_residuos(residuals, df):
    """Executar testes estatísticos nos resíduos"""
    # Teste de Shapiro-Wilk
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    shapiro_results = pd.DataFrame({
        'Teste': ["Shapiro-Wilk"],
        'Estatística': [shapiro_stat],
        'P-valor': [shapiro_p]
    })
    
    # Teste de Cochran
    grouped_variances = df.groupby('Nível')['Área'].var()
    max_variance = grouped_variances.max()
    sum_variances = grouped_variances.sum()
    cochran_c = max_variance / sum_variances
    
    cochran_results = pd.DataFrame({
        'Teste': ["Cochran C"],
        'Estatística': [cochran_c],
        'Observação': ["Consulte tabelas críticas para interpretação formal"]
    })
    
    return shapiro_results, shapiro_p, cochran_results, cochran_c

def interpretar_normalidade(shapiro_p):
    """Interpretar teste de normalidade"""
    if shapiro_p > ALPHA:
        st.success(f"P-valor ({shapiro_p:.4f}) > {ALPHA}: Não rejeitamos H0. Resíduos seguem distribuição normal.")
    else:
        st.warning(f"P-valor ({shapiro_p:.4f}) ≤ {ALPHA}: Rejeitamos H0. Resíduos podem não seguir distribuição normal.")

def criar_resumo_conclusoes(p_value_concentration, intercept_p_value, correlation_coefficient, shapiro_p):
    """Criar tabela resumo das conclusões"""
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
            'Modelo estatisticamente significativo' if p_value_concentration < ALPHA else 'Modelo não significativo',
            'Intercepto estatisticamente significativo' if intercept_p_value < ALPHA else 'Intercepto não significativo',
            'Relação linear adequada' if correlation_coefficient > CORRELATION_THRESHOLD else 'Relação linear questionável',
            'Resíduos seguem distribuição normal' if shapiro_p > ALPHA else 'Resíduos podem não ser normais',
            'Consultar valores críticos para conclusão formal',
            'Verificar gráficos de resíduos padronizados (valores > |3| são outliers)'
        ]
    }
    return pd.DataFrame(conclusions_data)

def criar_sidebar_info(model, r_squared, correlation_coefficient, p_value_concentration):
    """Criar informações na sidebar"""
    intercept = model.params['Intercept']
    slope = model.params['Q("Concentração mg/mL")']
    
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

# ===== APLICAÇÃO PRINCIPAL =====

# Título da aplicação
st.title('Linearidade para Validação Analítica')

st.write('''#### Realizar testes estatísticos da linearidade da Validação Analítica de acordo com a RDC 166.

Este é um requisito para os métodos que foram desenvolvidos e estão sendo utilizados para liberação de resultados.
''')

# Entrada de dados
dados = criar_interface_entrada()

# Criar DataFrame
df_cabeçalho = pd.DataFrame(dados, columns=COLUNAS_DF)

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

# Criar modelo e executar ANOVA
model_sm = criar_modelo_regressao(df_cabeçalho)
anova_summary = executar_anova(model_sm)

st.subheader("Tabela 2 - ANOVA")
st.dataframe(anova_summary)

# Interpretar ANOVA
p_value_concentration = interpretar_anova(anova_summary)

# Analisar coeficientes
coefficients_table, confidence_interval_table = analisar_coeficientes(model_sm)

st.subheader("Tabela 3 - Coeficientes")
st.dataframe(coefficients_table)

st.subheader("Tabela 4 - Intervalos de Confiança")
st.dataframe(confidence_interval_table)

# Teste do intercepto
st.subheader("3.2 - Teste do Intercepto")
intercept_p_value = testar_intercepto(coefficients_table)

# Qualidade do ajuste
st.subheader("3.3 - Qualidade do Ajuste")
summary_of_fit, correlation_coefficient = calcular_qualidade_ajuste(df_cabeçalho)

st.subheader("Tabela 5 - Medidas Descritivas da Qualidade do Ajuste")
st.dataframe(summary_of_fit)

interpretar_correlacao(correlation_coefficient)

st.header("4 - Análise Gráfica")

st.subheader("4.1 - Diagrama de Dispersão")
fig_dispersao = criar_grafico_dispersao(df_cabeçalho, model_sm)
st.pyplot(fig_dispersao)

st.subheader("4.2 - Diagnóstico dos Resíduos")
fig_residuos = criar_diagnosticos_residuos(model_sm, df_cabeçalho)
st.pyplot(fig_residuos)

st.header("5 - Testes Estatísticos dos Resíduos")

# Executar testes nos resíduos
residuals = model_sm.resid
shapiro_results, shapiro_p, cochran_results, cochran_c = executar_testes_residuos(residuals, df_cabeçalho)

st.subheader("5.1 - Teste de Normalidade (Shapiro-Wilk)")
st.dataframe(shapiro_results)
interpretar_normalidade(shapiro_p)

st.subheader("5.2 - Teste de Homocedasticidade (Cochran)")
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
summary_table = criar_resumo_conclusoes(p_value_concentration, intercept_p_value, correlation_coefficient, shapiro_p)
st.dataframe(summary_table)

# Calcular R² para a sidebar
X = df_cabeçalho[['Concentração mg/mL']]
y = df_cabeçalho['Área']
model_sklearn = LinearRegression()
model_sklearn.fit(X, y)
r_squared = model_sklearn.score(X, y)

# Informações na sidebar
criar_sidebar_info(model_sm, r_squared, correlation_coefficient, p_value_concentration)

# Rodapé
st.markdown("---")
st.write("**Nota:** Este aplicativo é para fins educacionais e de demonstração. Para análises formais, consulte um estatístico.")
