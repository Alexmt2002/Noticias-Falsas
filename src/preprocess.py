import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')


# Quita las palabras que no tienen importancia
stop_words = set(stopwords.words('english'))
# Pone las palabras en su forma base
lemmatizer = WordNetLemmatizer()

ruta = os.path.join( os.path.dirname(__file__),"..", "data", "WELFake_Dataset.csv")

df = pd.read_csv(ruta)

print(f"EL nombre de las columnas son: \n {df.columns}")
df = df.drop(columns=['Unnamed: 0'])


def limpiar_texto(texto):
    
    # pasar a minúsculas
    texto = texto.lower()

    # eliminar URLs
    texto = re.sub(r"http\S+|www\S+", "", texto)

    # eliminar HTML
    texto = re.sub(r"<.*?>", "", texto)

    # quitar puntuación y caracteres raros
    texto = re.sub(r"[^a-zA-Z\s]", "", texto)

    token = word_tokenize(texto)

    token = [palabra for palabra in token if palabra not in stop_words]

    token = [lemmatizer.lemmatize(palabra) for palabra in token]

    # volver a unir texto
    texto = " ".join(token)

    return texto


def limpiar_dataset(dataset):
    # 1. Eliminar nulos iniciales y duplicados
    dataset = dataset.dropna().drop_duplicates().copy()

    columnas_texto = ["title", "text"]
    for col in columnas_texto:
        dataset[col] = dataset[col].apply(limpiar_texto)

    # 2. Reemplazar strings vacíos (que dejó la limpieza) por NaN
    import numpy as np
    dataset.replace("", np.nan, inplace=True)
    
    # 3. Eliminar ahora los nuevos nulos creados
    dataset = dataset.dropna(subset=columnas_texto)

    return dataset

def guardar_dataset(dataset):

    ruta_guardado = os.path.join( os.path.dirname(__file__), "..",   "data",  "dataset_limpio.csv")

    dataset.to_csv(ruta_guardado, index=False)

    print("Dataset guardado correctamente")
    


# Cargar
dataset_limpio = limpiar_dataset(df)

# Guardar
guardar_dataset(dataset_limpio)

print(dataset_limpio.head())




