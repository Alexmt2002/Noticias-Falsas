from sentence_transformers import (
    SentenceTransformer,
    util
)


# =========================
# MODELO SEMÁNTICO
# =========================

modelo_similitud = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================
# CALCULAR SIMILITUD
# =========================

def calcular_similitud(
    texto1,
    texto2
):

    if not texto1 or not texto2:

        return 0

    embedding1 = modelo_similitud.encode(
        texto1,
        convert_to_tensor=True
    )

    embedding2 = modelo_similitud.encode(
        texto2,
        convert_to_tensor=True
    )

    similitud = util.cos_sim(
        embedding1,
        embedding2
    )

    score = similitud.item()

    return round(score * 100, 2)