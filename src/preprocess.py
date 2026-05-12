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
print("--------------")


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

    # eliminar filas vacías
    dataset_limpio = dataset.dropna().copy()

    # eliminar nulos
    dataset_limpio = dataset.dropna()

    # eliminar duplicados
    dataset_limpio = dataset_limpio.drop_duplicates()

    # limpiar columnas de texto
    columnas_texto = ["title", "text"]

    for col in columnas_texto:
        dataset_limpio[col] = dataset_limpio[col].apply(limpiar_texto)

    return dataset_limpio

def guardar_dataset(dataset):

    ruta_guardado = os.path.join( os.path.dirname(__file__), "..",   "data",  "dataset_limpio.csv")

    dataset.to_csv(ruta_guardado, index=False)

    print("Dataset guardado correctamente")
    


# Cargar
dataset_limpio = limpiar_dataset(df)

# Guardar
guardar_dataset(dataset_limpio)

print(dataset_limpio.head())


