import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# =========================
# CARGA DEL DATASET
# =========================

ruta = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "dataset_limpio.csv"
)

df = pd.read_csv(ruta)


# =========================
# PREPROCESAMIENTO
# =========================

df["content"] = df["title"] + " " + df["text"]

X = df["content"]
y = df["label"]


# =========================
# DIVISIÓN DE DATOS
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# VECTORIZACIÓN TF-IDF
# =========================

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# =========================
# ENTRENAMIENTO DEL MODELO
# =========================

modelo = LogisticRegression()

modelo.fit(X_train_tfidf, y_train)


# =========================
# PREDICCIONES Y MÉTRICAS
# =========================

predicciones = modelo.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predicciones)

print(f"Accuracy: {accuracy}")
print(classification_report(y_test, predicciones))


# =========================
# CREAR RUTA DE MODELOS
# =========================

ruta_model = os.path.join(
    os.path.dirname(__file__),
    "model"
)

os.makedirs(ruta_model, exist_ok=True)


# =========================
# GUARDAR MODELO Y VECTORIZADOR
# =========================

joblib.dump(
    modelo,
    os.path.join(ruta_model, "modelo_logreg.pkl")
)

joblib.dump(
    vectorizer,
    os.path.join(ruta_model, "vectorizador_logreg.pkl")
)