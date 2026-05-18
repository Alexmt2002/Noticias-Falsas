import joblib

from deep_translator import GoogleTranslator

from src.preprocess import limpiar_texto


# =========================
# CARGAR MODELOS
# =========================

print("Cargando modelos...")

modelo = joblib.load(
    "model/modelo_fake_news.pkl"
)

modelo_embedding = joblib.load(
    "model/modelo_embeddings.pkl"
)

print("Modelos cargados correctamente")


# comprobar clases
print("\nClases del modelo:")
print(modelo.classes_)


# =========================
# FUNCIÓN PRINCIPAL
# =========================

def predecir_noticia(texto):

    # validar texto
    if not texto or not texto.strip():

        return "ERROR: Debes introducir texto."

    try:

        # traducir a inglés
        try:

            texto_traducido = GoogleTranslator(
                source="auto",
                target="en"
            ).translate(texto)

        except:

            texto_traducido = texto

        # limpiar texto
        texto_limpio = limpiar_texto(
            texto_traducido
        )

        # generar embedding
        embedding = modelo_embedding.encode(
            [texto_limpio],
            normalize_embeddings=True
        )

        # predicción
        prediccion = modelo.predict(
            embedding
        )[0]

        # probabilidades
        probabilidades = modelo.predict_proba(
            embedding
        )[0]

        # resultado
        if prediccion == 1:

            confianza = probabilidades[1] * 100

            return (
                f"\nFAKE NEWS\n"
                f"Confianza: {confianza:.2f}%"
            )

        else:

            confianza = probabilidades[0] * 100

            return (
                f"\nREAL NEWS\n"
                f"Confianza: {confianza:.2f}%"
            )

    except Exception as e:

        return f"ERROR: {str(e)}"


# =========================
# CONSOLA
# =========================

if __name__ == "__main__":

    while True:

        print("\n========================")
        print(" DETECTOR DE FAKE NEWS ")
        print("========================")

        noticia = input(
            "\nIntroduce una noticia:\n\n> "
        )

        if noticia.lower() == "salir":

            print("\nCerrando programa...")
            break

        resultado = predecir_noticia(
            noticia
        )

        print("\n========================")
        print(" RESULTADO ")
        print("========================")

        print(resultado)