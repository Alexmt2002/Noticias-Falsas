import streamlit as st
import time

from src import predict, utils

from src.predict import predecir_noticia

from src.utils import (
    print_titulo,
    recortar_texto
)


# =========================
# CONFIGURACIÓN
# =========================

st.set_page_config(
    page_title="Fake News Detector AI",
    page_icon="📰",
    layout="centered"
)


# =========================
# ESTILO
# =========================

st.markdown("""
    <style>

    .main {
        padding-top: 2rem;
    }

    .titulo {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 10px;
    }

    .subtitulo {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 40px;
    }

    .resultado-box {
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        font-size: 20px;
        font-weight: bold;
    }

    </style>
""", unsafe_allow_html=True)


# =========================
# TÍTULO
# =========================

st.markdown(
    '<div class="titulo">📰 Fake News Detector AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitulo">'
    'Detección inteligente de noticias falsas usando NLP'
    '</div>',
    unsafe_allow_html=True
)


# =========================
# INPUT
# =========================

texto_usuario = st.text_area(
    "Introduce una noticia",
    height=220,
    placeholder=(
        "Ejemplo:\n\n"
        "Donald Trump was the president "
        "of the United States in 2017."
    )
)


# =========================
# BOTÓN
# =========================

if st.button("🔍 Analizar noticia"):

    if not texto_usuario.strip():

        st.warning(
            "Debes introducir una noticia."
        )

    else:

        with st.spinner(
            "Analizando noticia..."
        ):

            inicio = time.time()

            resultado = predecir_noticia(
                texto_usuario
            )

            tiempo = time.time() - inicio

        # =========================
        # RESULTADO
        # =========================

        st.markdown("## Resultado")

        if "FAKE NEWS" in resultado:

            st.error(resultado)

        elif "REAL NEWS" in resultado:

            st.success(resultado)

        else:

            st.warning(resultado)

        # =========================
        # INFORMACIÓN EXTRA
        # =========================

        st.markdown("### Información")

        st.write(
            f"⏱️ Tiempo de análisis: "
            f"{tiempo:.2f} segundos"
        )

        st.write(
            f"📝 Longitud del texto: "
            f"{len(texto_usuario)} caracteres"
        )

        st.write(
            f"📄 Vista previa:"
        )

        st.info(
            recortar_texto(
                texto_usuario,
                max_chars=300
            )
        )


# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Proyecto de detección de fake news "
    "usando Sentence Transformers y NLP"
)