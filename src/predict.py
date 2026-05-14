import os
import joblib

from preprocess import limpiar_texto


# =========================
# CARGAR MODELO Y TF-IDF
# =========================

ruta_modelo = os.path.join(
    os.path.dirname(__file__),
    "model",
    "modelo_logreg.pkl"
)

ruta_vectorizer = os.path.join(
    os.path.dirname(__file__),
    "model",
    "vectorizador_logreg.pkl"
)

modelo = joblib.load(ruta_modelo)
vectorizer = joblib.load(ruta_vectorizer)



def predecir_noticia(texto):

    # limpiar texto
    texto_limpio = limpiar_texto(texto)

    # convertir a vector TF-IDF
    vector = vectorizer.transform([texto_limpio])

    # hacer predicción
    prediccion = modelo.predict(vector)

    # convertir resultado a texto
    if prediccion[0] == 1:
        return "FAKE NEWS"
    else:
        return "REAL NEWS"


if __name__ == "__main__":

    noticia = input("Introduce una noticia: ")

    resultado = predecir_noticia(noticia)

    print("\nResultado:")
    print(resultado)