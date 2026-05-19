import spacy


# =========================
# MODELO NLP
# =========================

nlp = spacy.load(
    "en_core_web_sm"
)


# =========================
# EXTRAER ENTIDADES
# =========================

def extraer_entidades(texto):

    doc = nlp(texto)

    entidades = []

    for ent in doc.ents:

        entidades.append({
            "texto": ent.text,
            "tipo": ent.label_
        })

    return entidades