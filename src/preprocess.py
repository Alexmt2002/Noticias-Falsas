import pandas as pd
import numpy as np
import os
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')



# Quita las palabras que no tienen importancia
stop_words = set(stopwords.words('english'))

stop_words_en = set(stopwords.words('english'))
stop_words_es = set(stopwords.words('spanish'))

stop_words = stop_words_en.union(stop_words_es)



def limpiar_texto(texto):

    if not isinstance(texto, str):
        return ""

    texto = texto.lower()

    texto = re.sub(
        r"http\S+|www\S+",
        "",
        texto
    )

    texto = re.sub(
        r"<.*?>",
        "",
        texto
    )

    texto = re.sub(
        r"[^\w\s]",
        "",
        texto
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto
    ).strip()

    return texto

def limpiar_dataset(dataset):

    dataset = dataset.dropna().drop_duplicates().copy()

    columnas_texto = ["title", "full_text"] #Importante dependiendo del dataset

    for col in columnas_texto:

        dataset[col] = dataset[col].astype(str)

        dataset[col] = dataset[col].apply(
            limpiar_texto
        )

    # eliminar vacíos
    dataset.replace("", np.nan, inplace=True)

    dataset.dropna(
        subset=columnas_texto,
        inplace=True
    )

    return dataset

def guardar_dataset(dataset):

    ruta_guardado = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "dataset_limpio.csv"
    )

    dataset.to_csv(
        ruta_guardado,
        index=False,
        encoding="utf-8"
    )

    print("Dataset guardado correctamente")


if __name__ == "__main__":

    ruta = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "GlobalFakeNews_Research2026_v1.csv"
    )

    df = pd.read_csv(ruta)

    print(f"Columnas:\n{df.columns}")

    # eliminar columna innecesaria
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # limpiar
    dataset_limpio = limpiar_dataset(df)

    # guardar
    guardar_dataset(dataset_limpio)

    print(dataset_limpio.head())





