# Aplicativo Streamlit - Análise de Linearidade

Este aplicativo foi adaptado para funcionar com Streamlit para análise de linearidade em validação analítica.

## Como executar:

1. **Instalar as dependências:**
```bash
pip install -r requirements.txt
```

2. **Executar o aplicativo:**
```bash
streamlit run linearidade.py
```

## Alternativa - Usar o arquivo separado:

Se houver problemas com o arquivo original, use:
```bash
streamlit run linearidade_streamlit.py
```

## Dependências necessárias:

- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scipy
- scikit-learn

## Funcionalidades:

- Análise de linearidade automática
- Gráficos interativos
- Testes estatísticos (ANOVA, Shapiro-Wilk)
- Diagnóstico de resíduos
- Resumo das conclusões

## Problemas comuns:

1. **Imports não encontrados:** Execute `pip install -r requirements.txt`
2. **Erro ao executar:** Verifique se está usando Python 3.7+
3. **Gráficos não aparecem:** Certifique-se que está executando via Streamlit

## Interface:

- Dados de exemplo pré-carregados
- Sidebar com informações do modelo
- Resultados organizados por seções
- Gráficos de diagnóstico automáticos

O aplicativo processa automaticamente os dados e apresenta:
- Tabelas estatísticas
- Gráficos de dispersão e resíduos
- Testes de hipóteses
- Conclusões sobre linearidade
