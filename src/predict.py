from IPython import terminal
from IPython import terminal
from IPython import terminal
import joblib

from deep_translator import GoogleTranslator
from preprocess import limpiar_texto

from classify_topic import clasificar_tema
from fact_check import verificar_wikipedia

from similarity import calcular_similitud




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

            tema, confianza_tema = clasificar_tema(
                texto
            )

            print(f"\nTema detectado: {tema}")

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
        evidencia = verificar_wikipedia(
            texto_traducido
        )

        if evidencia:

            print("\nEvidencia encontrada:")

            print(evidencia[:300])
        
        # calcular similitud
        similitud = calcular_similitud(
            texto_traducido,
            evidencia
        )

        print(
            f"\nSimilitud semántica: "
            f"{similitud}%"
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
        # score NLP
        if prediccion == 1:

            confianza_nlp = probabilidades[1] * 100

        else:

            confianza_nlp = probabilidades[0] * 100


        # score híbrido
        score_final = (
            (confianza_nlp * 0.6) +
            (similitud * 0.4)
        )


        # decisión final
        if score_final >= 60:

            resultado_final = "REAL NEWS"

        else:

            resultado_final = "FAKE NEWS"


        return (
            f"\n{resultado_final}\n"
            f"Score final: {score_final:.2f}%\n"
            f"Tema: {tema}\n"
            f"Similitud: {similitud}%"
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