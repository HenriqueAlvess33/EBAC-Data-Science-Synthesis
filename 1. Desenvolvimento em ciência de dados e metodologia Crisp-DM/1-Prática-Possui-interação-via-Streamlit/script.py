import streamlit as st


def main():
    # Configura o título da aplicação
    st.set_page_config(
        page_title="Previsão de renda dos clientes",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="varig_icon.png",
    )
    st.markdown(
        """
    <link href="https://fonts.googleapis.com/css2?family=Kantumruy+Pro&display=swap" rel="stylesheet">

    <h1 style='text-align: center; font-family: "Kantumruy Pro", sans-serif; font-size: 3em;'>
        <strong>Atividades Introdutórias ao Python</strong>
    </h1>
    """,
        unsafe_allow_html=True,
    )


main()

aba1, aba2, aba3, aba4 = st.tabs(
    ["💊 Gravidez", "➗ Divisibilidade", "🔢 Números Primos", "⚖️ Calculadora de IMC"]
)
with aba1:
    with st.expander("🧪 Teste de Gravidez - Beta hCG", expanded=False):

        # Seção de gênero
        genero = st.radio(
            "**👥 Gênero**",
            options=["Masculino", "Feminino"],
            index=0,
            horizontal=True,
            key="genero_radio",
        )

        # Seção do exame
        st.subheader("📊 Resultado do Exame")
        beta_hcg = st.number_input(
            "Nível de beta-hCG (mUI/mL):",
            min_value=0.0,
            max_value=1000.0,
            value=0.0,
            step=0.1,
            help="Valor normal: < 5 mUI/mL",
        )

        # Interpretação médica
        st.subheader("🔍 Interpretação")

        if beta_hcg < 5:
            st.success("✅ **Resultado Negativo**")
            st.info("Nível de beta-hCG dentro da normalidade")

        elif 5 <= beta_hcg <= 25:
            st.warning("⚠️ **Resultado Indeterminado**")
            st.info("Pode ser muito precoce. Repita o teste em 48-72 horas.")

        else:
            st.error("🎯 **Resultado Positivo**")

            if genero == "Feminino":
                # Estimativa gestacional baseada em beta-hCG
                if beta_hcg < 1000:
                    st.success("📅 Provavelmente 3-4 semanas de gestação")
                elif 1000 <= beta_hcg < 10000:
                    st.success("📅 Provavelmente 4-5 semanas de gestação")
                else:
                    st.success("📅 Provavelmente mais de 5 semanas")

                st.info("🩺 Procure acompanhamento médico para confirmação")

            else:
                st.error("🚨 **ATENÇÃO MÉDICA URGENTE**")
                st.warning(
                    """
                Níveis elevados de beta-hCG em homens podem indicar:
                - Tumores germinativos
                - Outras condições médicas
                **Procure um médico imediatamente!**
                """
                )

    # Informações adicionais
    with st.expander("ℹ️ Informações sobre beta-hCG"):
        st.write(
            """
            - **Não grávida:** < 5 mUI/mL
            - **3 semanas:** 5-50 mUI/mL
            - **4 semanas:** 5-426 mUI/mL  
            - **5 semanas:** 18-7,340 mUI/mL
            - **6 semanas:** 1,080-56,500 mUI/mL
            - **Homens:** < 5 mUI/mL (valores elevados requerem investigação)
            """
        )
with aba2:
    with st.expander("É divisível ?"):
        # Nota: Vamos usar 'min_value' para evitar entradas negativas, se desejar.
        # O passo crucial: definir min_value=1 para M para evitar o zero completamente!
        N = st.number_input(
            "Selecione um valor para ser dividido (Numerador)", step=1, value=10
        )
        M = st.number_input(
            "Selecione um valor para dividir (Denominador)",
            step=1,
            value=2,
            min_value=1,
        )  # <-- A SOLUÇÃO PRINCIPAL

        st.caption(f"Testando se {N} é divisível por {M}...")

        # Lógica Impecável
        if M == 0:
            # Esta condição agora é quase impossível devido ao min_value=1,
            # mas é uma boa prática defensiva mantê-la.
            st.error(
                "❌ Erro Crítico: O denominador (valor para dividir) não pode ser zero."
            )
        else:
            # Agora podemos calcular com segurança
            if N % M == 0:
                st.success(
                    f"✅ Sim! {N} é perfeitamente divisível por {M}. O resultado é {N // M}."
                )
            else:
                st.warning(
                    f"⚠️ Não. {N} **não** é divisível por {M}. O resto da divisão é {N % M}."
                )

with aba3:
    with st.expander("🔍 É primo?", expanded=True):

        st.subheader("Verificação de Número Primo")

        N = st.number_input(
            "Digite um número inteiro maior ou igual a 2:",
            min_value=2,
            step=1,
            value=17,
            key="verificador_primo",
        )

        st.caption(f"Verificando se {N} é um número primo...")
        st.caption("_Um número primo é divisível apenas por 1 e por ele mesmo._")

        eh_primo = True

        if N > 2 and N % 2 == 0:
            eh_primo = False
            divisor = 2  #
        else:

            i = 3
            while i * i <= N:
                if N % i == 0:
                    eh_primo = False
                    divisor = i
                    break
                i += 2

        st.subheader("Resultado:")

        if eh_primo:
            st.success(f"✅ **{N} é um número primo!**")
            st.info("É divisível apenas por 1 e por ele mesmo.")
        else:

            st.error(f"❌ **{N} não é um número primo.**")
            st.write(f"É divisível por **{divisor}** (e outros).")
with aba4:
    with st.expander("📋 Verificador de IMC", expanded=True):  # Adicionei um emoji
        # Seu dicionário está PERFEITO. Não mexa nele.
        informacoes_imc = {
            "Abaixo do peso": {
                "intervalo": "Menor que 18,5",
                "descricao": "Indica que a pessoa está abaixo do peso considerado saudável. Pode estar associado a desnutrição, deficiências nutricionais ou outras condições médicas.",
                "recomendacao": "Buscar avaliação médica e nutricional para entender a causa e planejar a recuperação de peso de forma saudável.",
            },
            "Peso normal": {
                "intervalo": "18,5 – 24,9",
                "descricao": "Faixa considerada saudável, associada a menor risco de doenças relacionadas ao peso.",
                "recomendacao": "Manter hábitos equilibrados de alimentação, prática regular de atividade física e acompanhamento periódico da saúde.",
            },
            "Sobrepeso": {
                "intervalo": "25,0 – 29,9",
                "descricao": "Indica acúmulo de peso acima do recomendado. Pode aumentar o risco de problemas como hipertensão e diabetes.",
                "recomendacao": "Adotar mudanças no estilo de vida com alimentação balanceada e aumento da prática de exercícios físicos.",
            },
            "Obesidade grau I": {
                "intervalo": "30,0 – 34,9",
                "descricao": "Primeiro nível de obesidade, já considerado fator de risco para doenças cardiovasculares, diabetes tipo 2 e problemas articulares.",
                "recomendacao": "Procurar acompanhamento médico e nutricional para elaborar um plano de controle de peso.",
            },
            "Obesidade grau II": {
                "intervalo": "35,0 – 39,9",
                "descricao": "Obesidade severa, com risco elevado de doenças graves como apneia do sono, doenças cardíacas e metabólicas.",
                "recomendacao": "Requer acompanhamento profissional multidisciplinar e mudanças intensivas no estilo de vida.",
            },
            "Obesidade grau III": {
                "intervalo": "Maior ou igual a 40,0",
                "descricao": "Também chamada de obesidade mórbida, está associada a risco muito elevado de complicações graves e redução da expectativa de vida.",
                "recomendacao": "Necessário acompanhamento médico rigoroso; em alguns casos, pode ser indicada cirurgia bariátrica.",
            },
        }

        st.subheader("Insira seus dados")
        col1, col2 = st.columns(2)
        with col1:
            altura = st.number_input(
                "Altura (metros):", step=0.01, min_value=1.00, value=1.70
            )
        with col2:
            peso = st.number_input("Peso (kg):", step=0.1, min_value=20.0, value=70.0)

        # Cálculo do IMC
        imc_calculado = round(peso / (altura**2), 2)

        # Determinar a classificação CORRETAMENTE
        if imc_calculado < 18.5:
            classificacao = "Abaixo do peso"
        elif 18.5 <= imc_calculado < 25:
            classificacao = "Peso normal"
        elif 25 <= imc_calculado < 30:
            classificacao = "Sobrepeso"
        elif 30 <= imc_calculado < 35:
            classificacao = "Obesidade grau I"
        elif 35 <= imc_calculado < 40:
            classificacao = "Obesidade grau II"
        else:  # imc_calculado >= 40
            classificacao = "Obesidade grau III"

        # Exibir resultados de forma organizada
        st.divider()
        st.subheader("Seu Resultado")

        # Usar uma métrica para mostrar o IMC
        st.metric(label="**Seu IMC é:**", value=f"{imc_calculado}")

        # Criar um container para o diagnóstico para dar destaque
        diagnostico_container = st.container(border=True)
        with diagnostico_container:
            st.markdown(f"**Classificação:** {classificacao}")
            st.markdown(
                f"*Faixa de IMC: {informacoes_imc[classificacao]['intervalo']}*"
            )
            st.write("")  # Espaçamento
            st.info(informacoes_imc[classificacao]["descricao"])
            st.write("")  # Espaçamento
            st.success(
                f"**Recomendação:** {informacoes_imc[classificacao]['recomendacao']}"
            )

        # Disclaimer importante
        st.caption(
            "_⚠️ Este cálculo fornece uma classificação geral. Para uma avaliação de saúde completa, consulte sempre um profissional médico. O IMC não considera fatores como massa muscular, idade, sexo ou genética._"
        )
