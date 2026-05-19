import wikipedia


# idioma wikipedia
wikipedia.set_lang("en")


# =========================
# BUSCAR EVIDENCIA
# =========================

def verificar_wikipedia(texto):

    try:

        resultado = wikipedia.summary(
            texto,
            sentences=2
        )

        return resultado

    except:

        return None