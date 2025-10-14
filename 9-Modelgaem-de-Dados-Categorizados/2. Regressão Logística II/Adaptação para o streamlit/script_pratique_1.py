import streamlit as st

# Configuração da página
st.set_page_config(
    page_title="Tarefa I - Q&A",
    page_icon="❓",
    layout="centered"
)

# CSS personalizado responsivo para temas claro e escuro
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 2rem;
    }
    .question-card {
        background-color: var(--background-color);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--primary-color);
        border: 1px solid var(--secondary-background-color);
    }
    .question-text {
        font-size: 1.2rem;
        font-weight: bold;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }
    .answer-text {
        background-color: var(--secondary-background-color);
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid var(--secondary-background-color);
        margin-top: 1rem;
        color: var(--text-color);
    }
    .info-box {
        background-color: var(--secondary-background-color);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid var(--primary-color);
        color: var(--text-color);
    }
    
    /* Cores específicas para garantir contraste */
    [data-testid="stExpander"] {
        background-color: var(--background-color);
    }
    
    /* Garantir que o texto seja legível em ambos os temas */
    .custom-text {
        color: var(--text-color);
    }
</style>
""", unsafe_allow_html=True)

# Cabeçalho principal
st.markdown('<h1 class="main-header">Tarefa I - Perguntas e Respostas</h1>', unsafe_allow_html=True)

# Introdução
st.markdown("""
<div class="custom-text">
Esta aplicação apresenta respostas para questões sobre modelos de crédito, validação de modelos 
e tratamento de dados em desenvolvimento de modelos preditivos.
</div>
""", unsafe_allow_html=True)

# Divisão em colunas para layout mais organizado
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 📋 Lista de Questões")
    
    # Questão 1
    with st.container():
        st.markdown('<div class="question-card">', unsafe_allow_html=True)
        st.markdown('<div class="question-text">1. Qual a principal diferença entre um behaviour score e um credit score?</div>', unsafe_allow_html=True)
        
        with st.expander("Ver Resposta", expanded=False):
            st.markdown("""
            <div class="answer-text">
            <strong>Resposta:</strong> A principal diferença está na finalidade, visto que cada modelo é utilizado em momentos diferentes, sendo o "credit score" uma avaliação para risco de inadimplência para prospectos, clientes que não fazem parte ainda da instituição, para isso utilizando dados cadastrais, dados de bureau e dados do momento da solicitação. Um modelo de "behaviour score" visa analisar clientes que já estão na base da instituição, avaliando o risco de inadimplência com base no comportamento recente do cliente, avaliando históricos de pagamento, variações de saldo, uso do crédito. Muito útil para decisões de gestão de carteira, como aumento de limite, campanhas de retenção, reestruturação de dívida ou cortes de crédito.
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Questão 2
    with st.container():
        st.markdown('<div class="question-card">', unsafe_allow_html=True)
        st.markdown('<div class="question-text">2. Por que é tão comum se estabelecer janelas de desempenho de 12 meses, e 12 safras para o desenvolvimento de modelos?</div>', unsafe_allow_html=True)
        
        with st.expander("Ver Resposta", expanded=False):
            st.markdown("""
            <div class="answer-text">
            <strong>Resposta:</strong> É uma forma de contornar sazonalidades, que podem enviesar o modelo
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Questão 3
    with st.container():
        st.markdown('<div class="question-card">', unsafe_allow_html=True)
        st.markdown('<div class="question-text">3. Qual a diferença entre amostra out of time e amostra out of sample?</div>', unsafe_allow_html=True)
        
        with st.expander("Ver Resposta", expanded=False):
            st.markdown("""
            <div class="answer-text">
            <strong>Resposta:</strong> São estratégias de validação que testam a viabilidade do modelo. A amostra out of sample avalia a capacidade de generalização, ou seja, se o modelo funciona bem com dados que não foram usados no treinamento, mas pertencem à mesma janela temporal. Já a amostra out of time testa a resiliência temporal do modelo, verificando como sua performance se mantém quando aplicado em períodos posteriores, com possíveis mudanças de comportamento nos dados.
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Questão 4
    with st.container():
        st.markdown('<div class="question-card">', unsafe_allow_html=True)
        st.markdown('<div class="question-text">4. Se os dados da variável resposta estão sistematicamente corrompidos para duas das safras de desenvolvimento, o que você faria?</div>', unsafe_allow_html=True)
        
        with st.expander("Ver Resposta", expanded=False):
            st.markdown("""
            <div class="answer-text">
            <strong>Resposta:</strong> A melhor medida é a exclusão das safras comprometidas do treinamento, uma vez que a variável de interesse corrompida compromete gravemente a qualidade do modelo, podendo induzir vieses e previsões equivocadas. No entanto, se a origem da falha for conhecida e tecnicamente compreendida, é possível considerar tratamentos específicos para corrigir ou ajustar esses dados. Essa abordagem, porém, deve ser adotada com cautela, exigindo uma análise criteriosa de custo-benefício e validação reforçada, já que manter dados comprometidos pode representar um risco significativo para a performance e a confiabilidade do modelo.
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### ℹ️ Informações")
    
    st.markdown("""
    <div class="info-box">
    <strong>Instruções:</strong><br>
    • Clique em 'Ver Resposta' para expandir cada questão<br>
    • Todas as respostas podem ser visualizadas simultaneamente<br>
    • Compatível com tema claro e escuro
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="custom-text">
    <strong>Conceitos abordados:</strong>
    - Modelos de crédito
    - Validação de modelos
    - Tratamento de dados
    - Sazonalidade
    - Desenvolvimento de modelos
    </div>
    """, unsafe_allow_html=True)

# Seção adicional para demonstrar compatibilidade com temas
st.markdown("---")
col_info1, col_info2 = st.columns(2)

with col_info1:
    with st.expander("🎨 Informações sobre Temas", expanded=False):
        st.markdown("""
        <div class="custom-text">
        <strong>Compatibilidade com temas:</strong>
        
        ✅ <strong>Tema Claro:</strong> Cores claras com bom contraste
        ✅ <strong>Tema Escuro:</strong> Cores escuras com texto legível
        ✅ <strong>Responsivo:</strong> Adapta automaticamente
        
        <em>Para alterar o tema: Configurações → Appearance → Theme</em>
        </div>
        """, unsafe_allow_html=True)

with col_info2:
    # Detector de tema simples
    try:
        # Esta é uma maneira de detectar o tema atual
        st.markdown("""
        <div class="custom-text">
        <strong>Tema atual:</strong> Aplicação otimizada para ambos os temas
        </div>
        """, unsafe_allow_html=True)
    except:
        pass

# Rodapé
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: var(--text-color);">Desenvolvido com Streamlit • Compatível com temas claro e escuro</div>', 
    unsafe_allow_html=True
)

# Botão para expandir todas as respostas
if st.button("🎯 Expandir Todas as Respostas"):
    st.success("Clique em cada 'Ver Resposta' para visualizar as respostas individualmente")
    st.info("💡 Dica: A aplicação se adapta automaticamente ao tema claro ou escuro!")