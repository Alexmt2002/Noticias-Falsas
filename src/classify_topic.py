from transformers import pipeline


# =========================
# CLASIFICADOR DE TEMAS
# =========================

clasificador = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)


# =========================
# TEMAS DISPONIBLES
# =========================

TEMAS = [
    "politics",
    "sports",
    "science",
    "history",
    "technology",
    "entertainment"
]


# =========================
# FUNCIÓN PRINCIPAL
# =========================

def clasificar_tema(texto):

    resultado = clasificador(
        texto,
        TEMAS
    )

    tema = resultado["labels"][0]

    confianza = resultado["scores"][0]

    return tema, confianza