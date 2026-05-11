import pandas as pd
import os
import re

ruta = os.path.join(os.getcwd(), "WELFake_Dataset.csv")

df = pd.read_csv(ruta)

print(f"EL nombre de las columnas son: \n {df.columns}")
df = df.drop(columns=['Unnamed: 0'])
print("--------------")


def limpiar_texto(texto):
    
    # pasar a minúsculas
    texto = texto.lower()

    # eliminar URLs
    texto = re.sub(r"http\S+|www\S+", "", texto)

    # eliminar HTML
    texto = re.sub(r"<.*?>", "", texto)

    # eliminar caracteres raros y emojis
    texto = re.sub(r"[^a-zA-Záéíóúñü\s]", "", texto)

    # eliminar espacios duplicados
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def limpiar_dataset(dataset):

    # eliminar filas vacías
    dataset_limpio = dataset.dropna().copy()

    # eliminar duplicados
    dataset_limpio = dataset_limpio.drop_duplicates()

    # limpiar columnas de texto
    columnas_texto = ["title", "text"]

    for col in columnas_texto:
        dataset_limpio[col] = dataset_limpio[col].apply(limpiar_texto)

    return dataset_limpio


dataset_limpio = limpiar_dataset(df)

print(dataset_limpio.head())
print(dataset_limpio.info())