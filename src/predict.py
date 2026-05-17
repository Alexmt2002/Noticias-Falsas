import os
import joblib

from preprocess import limpiar_texto
from deep_translator import GoogleTranslator


# =========================
# CARGAR MODELO Y TF-IDF
# =========================

BASE_DIR = os.path.dirname(__file__)

ruta_modelo = os.path.join(
    BASE_DIR,
    "model",
    "modelo_logreg.pkl"
)

ruta_vectorizer = os.path.join(
    BASE_DIR,
    "model",
    "vectorizador_logreg.pkl"
)

print("Cargando modelo...")
modelo = joblib.load(ruta_modelo)
vectorizer = joblib.load(ruta_vectorizer)
print("Modelo cargado correctamente")



def predecir_noticia(texto):

    # Validar entrada
    if not texto or not texto.strip():

        return "ERROR: Debes introducir texto."

    try:

        # traducir automáticamente a inglés
        texto_traducido = GoogleTranslator(
            source='auto',
            target='en'
        ).translate(texto)

        # limpiar texto
        texto_limpio = limpiar_texto(
            texto_traducido
        )

        # convertir a vector
        vector = vectorizer.transform(
            [texto_limpio]
        )

        # predicción
        prediccion = modelo.predict(vector)[0]

        # probabilidades
        probabilidades = modelo.predict_proba(
            vector
        )[0]

        confianza_real = probabilidades[0] * 100

        confianza_fake = probabilidades[1] * 100

        # resultado final
        if prediccion == 1:

            return (
                f"\nFAKE NEWS\n"
                f"Confianza: {confianza_fake:.2f}%"
            )

        else:

            return (
                f"\nREAL NEWS\n"
                f"Confianza: {confianza_real:.2f}%"
            )

    except Exception as e:

        return f"ERROR: {str(e)}"
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

        

