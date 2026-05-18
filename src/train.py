import os
import joblib
import pandas as pd

from sentence_transformers import SentenceTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report,
    accuracy_score
)

from preprocess import limpiar_texto

from config import (
    DATASET_PATH,
    MODEL_NAME
)


# =========================
# CARGAR DATASET
# =========================

df = pd.read_csv(DATASET_PATH)

df = df.dropna()

df["title"] = df["title"].astype(str)
df["text"] = df["text"].astype(str)


# =========================
# CREAR CONTENIDO
# =========================

df["content"] = (
    df["title"] + " " + df["text"]
)

# limpiar texto
df["content"] = df["content"].apply(
    limpiar_texto
)


# =========================
# TRAIN / TEST
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    df["content"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)


# =========================
# MODELO DE EMBEDDINGS
# =========================

print("Cargando modelo NLP...")

modelo_embedding = SentenceTransformer(
    MODEL_NAME
)


# =========================
# GENERAR EMBEDDINGS
# =========================

print("Generando embeddings train...")

X_train_embeddings = modelo_embedding.encode(
    X_train.tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Generando embeddings test...")

X_test_embeddings = modelo_embedding.encode(
    X_test.tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)


# =========================
# ENTRENAR MODELO
# =========================

modelo = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

modelo.fit(
    X_train_embeddings,
    y_train
)


# =========================
# PREDICCIONES
# =========================

predicciones = modelo.predict(
    X_test_embeddings
)

accuracy = accuracy_score(
    y_test,
    predicciones
)

print(f"\nAccuracy: {accuracy:.4f}\n")

print(
    classification_report(
        y_test,
        predicciones
    )
)


# =========================
# GUARDAR MÉTRICAS
# =========================

os.makedirs("model", exist_ok=True)

with open("model/metricas.txt", "w") as f:

    f.write(f"Accuracy: {accuracy:.4f}\n\n")

    f.write(
        classification_report(
            y_test,
            predicciones
        )
    )


# =========================
# GUARDAR MODELOS
# =========================

joblib.dump(
    modelo,
    "model/modelo_fake_news.pkl"
)

joblib.dump(
    modelo_embedding,
    "model/modelo_embeddings.pkl"
)

print("\nModelos guardados correctamente")