def comparar_entidades(entidades_usuario, entidades_wiki):

    score = 0

    user_entities = {
        (e["texto"].lower(), e["tipo"])
        for e in entidades_usuario
    }

    wiki_entities = {
        (e["texto"].lower(), e["tipo"])
        for e in entidades_wiki
    }

    # coincidencias exactas
    for ent in user_entities:
        if ent in wiki_entities:
            score += 1

    # penalización por contradicción de fechas
    user_dates = [
        e["texto"] for e in entidades_usuario if e["tipo"] == "DATE"
    ]

    wiki_dates = [
        e["texto"] for e in entidades_wiki if e["tipo"] == "DATE"
    ]

    if user_dates and wiki_dates:

        # si NO coincide ninguna fecha → sospechoso
        if not any(d in wiki_dates for d in user_dates):
            score -= 2

    return score