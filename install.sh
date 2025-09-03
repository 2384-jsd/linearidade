#!/bin/bash
# Script de instalação das dependências

echo "🚀 Instalando dependências para o aplicativo de Linearidade..."

# Verificar se pip está disponível
if command -v pip3 &> /dev/null; then
    PIP=pip3
elif command -v pip &> /dev/null; then
    PIP=pip
else
    echo "❌ pip não encontrado. Instale Python e pip primeiro."
    exit 1
fi

echo "📦 Usando: $PIP"

# Instalar dependências
$PIP install streamlit pandas numpy matplotlib seaborn statsmodels scipy scikit-learn

echo "✅ Instalação concluída!"
echo ""
echo "🎯 Para executar o aplicativo:"
echo "   streamlit run linearidade.py"
echo ""
echo "📝 Ou use o arquivo limpo:"
echo "   streamlit run linearidade_streamlit.py"
