import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# =========================
# CARGA DEL DATASET
# =========================

ruta = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "dataset_limpio.csv"
)

df = pd.read_csv(
    ruta,
    encoding="utf-8"
)

print("\nDistribución de clases:")
print(df["label"].value_counts())

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
    random_state=42,
    stratify=y
)


# =========================
# VECTORIZACIÓN TF-IDF
# =========================

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# =========================
# ENTRENAMIENTO DEL MODELO
# =========================

modelo = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

modelo.fit(X_train_tfidf, y_train)


# =========================
# PREDICCIONES Y MÉTRICAS
# =========================

predicciones = modelo.predict(X_test_tfidf)

probabilidades = modelo.predict_proba(
    X_test_tfidf
)

accuracy = accuracy_score(
    y_test,
    predicciones
)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predicciones
    )
)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_test,
        predicciones
    )
)


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
    os.path.join(
        ruta_model,
        "modelo_logreg.pkl"
    ),
    compress=3
)

joblib.dump(
    vectorizer,
    os.path.join(
        ruta_model,
        "vectorizador_logreg.pkl"
    ),
    compress=3
)

feature_names = vectorizer.get_feature_names_out()
coeficientes = modelo.coef_[0]
top_fake = coeficientes.argsort()[-20:]
top_real = coeficientes.argsort()[:20]

print("\nPalabras más asociadas a FAKE NEWS:\n")

for i in reversed(top_fake):
    print(feature_names[i])

print("\nPalabras más asociadas a REAL NEWS:\n")

for i in top_real:
    print(feature_names[i])