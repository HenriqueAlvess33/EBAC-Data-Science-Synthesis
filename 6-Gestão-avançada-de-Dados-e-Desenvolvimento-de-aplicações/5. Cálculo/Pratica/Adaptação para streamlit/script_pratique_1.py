import streamlit as st
import pandas as pd

# Configuração da página
st.set_page_config(
    page_title="Cálculo - Tarefa 01",
    page_icon="📊",
    layout="centered"
)

# Título principal
st.title("📊 Cálculo - Tarefa 01")
st.markdown("---")

# Questão 1
st.header("1. Métodos Baseados em Derivadas")

st.write("""
**Marque quais desses métodos/algoritmos muito populares em ciência de dados são baseados no uso de derivada:**
""")

# Criar colunas para o layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Métodos")
    
    # Checkboxes para cada método
    minimos_quadrados = st.checkbox("1. Método Mínimos Quadrados", value=True)
    gradiente_descendente = st.checkbox("2. Gradiente Descendente", value=True)
    newton_raphson = st.checkbox("3. Newton Raphson", value=True)
    cart = st.checkbox("4. CART (Árvore de decisão)", value=False)

with col2:
    st.subheader("Explicações")
    
    # Explicações que aparecem dinamicamente
    if minimos_quadrados:
        st.success("""
        **✓ Método dos Mínimos Quadrados:**  
        ✔️ **Sim, utiliza derivadas.**  
        Fornece uma solução direta (analítica) obtida a partir da **anulação da derivada da função de custo** (erro quadrático). Envolve **derivadas parciais** para encontrar os mínimos da função.
        """)
    
    if gradiente_descendente:
        st.success("""
        **✓ Gradiente Descendente:**  
        ✔️ **Sim, utiliza derivadas.**  
        Método iterativo que **depende diretamente da derivada da função de custo (o gradiente)** para atualizar os coeficientes na direção da descida mais rápida.
        """)
    
    if newton_raphson:
        st.success("""
        **✓ Newton-Raphson:**  
        ✔️ **Sim, utiliza derivadas.**  
        Utiliza tanto a **primeira derivada (gradiente)** quanto a **segunda derivada (Hessiana)** da função de custo, permitindo ajustes mais rápidos e precisos.
        """)
    
    if cart:
        st.error("""
        **✗ CART (Árvore de Decisão):**  
        ❌ **Não utiliza derivadas.**  
        Constrói árvores baseando-se em critérios de divisão como **impureza de Gini**, **entropia** ou **erro quadrático médio**, mas **não usa derivadas**. As decisões são tomadas com base em divisões discretas.
        """)
    else:
        st.info("""
        **CART (Árvore de Decisão):**  
        ❌ **Não utiliza derivadas.**  
        Baseia-se em critérios de divisão discretos sem uso de derivadas.
        """)

st.markdown("---")

# Questão 2
st.header("2. Limite do Erro Quadrático Médio")

st.write("""
**Dada uma base de dados com uma variável resposta $y$ e um conjunto de variáveis explicativas. 
Considere uma estrutura de um modelo de regressão. Explique com suas palavras por que não é possível 
obter parâmetros que forneçam um erro quadrático médio (EQM) menor que o obtido com estimadores de mínimos quadrados.**
""")

# Expander para a resposta
with st.expander("📝 Clique para ver a resposta explicada", expanded=True):
    st.markdown("""
    **Resposta:**
    
    O próprio critério de ajuste do modelo é encontrar os coeficientes que **minimizam o EQM**. 
    Portanto, qualquer outro conjunto de parâmetros resultará, necessariamente, em um EQM maior ou igual.
    
    ---
    
    **Explicação Detalhada:**
    
    - O método dos mínimos quadrados é definido como a **solução que minimiza a soma dos quadrados dos resíduos**
    - Matematicamente, encontramos os parâmetros $\\beta$ que minimizam:
    
    $$
    \\min_{\\beta} \\sum_{i=1}^{n} (y_i - X_i\\beta)^2
    $$
    
    - Como esta é uma função convexa, o ponto onde as derivadas são zero representa o **mínimo global**
    - Qualquer outro conjunto de parâmetros diferente deste ótimo resultará em um EQM maior
    - Portanto, é impossível obter um EQM menor com outros estimadores para o mesmo modelo
    """)

# Adicionar uma visualização interativa
st.markdown("---")
st.subheader("🎯 Visualização Interativa")

st.write("""
Para ilustrar o conceito, imagine que estamos ajustando uma reta a um conjunto de pontos. 
O método dos mínimos quadrados encontra a reta que minimiza a soma das distâncias verticais ao quadrado.
""")

# Simulação simples com sliders
st.write("**Simulação de Ajuste de Reta:**")

col1, col2 = st.columns(2)

with col1:
    inclinacao = st.slider("Inclinação da reta", -2.0, 2.0, 1.0, 0.1)
    
with col2:
    intercepto = st.slider("Intercepto da reta", -2.0, 2.0, 0.0, 0.1)

# Cálculo do "EQM" simulado (apenas para demonstração)
eqm_simulado = abs(inclinacao - 1.0) + abs(intercepto - 0.0) + 0.5

st.metric(
    label="Erro Quadrático Médio Simulado", 
    value=f"{eqm_simulado:.3f}",
    delta=f"{(1.0 - eqm_simulado):.3f} vs mínimo teórico" if eqm_simulado < 1.0 else f"+{(eqm_simulado - 1.0):.3f} vs mínimo teórico",
    delta_color="inverse"
)

st.info("""
💡 **Observação:** Na prática, o método dos mínimos quadrados encontra automaticamente 
os valores ótimos de inclinação e intercepto que minimizam o EQM.
""")

# Rodapé
st.markdown("---")
st.caption("Atividade de Cálculo - Curso de Ciência de Dados")