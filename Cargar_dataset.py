import pandas as pd
import os

ruta = os.path.join(os.getcwd(), "WELFake_Dataset.csv")

df = pd.read_csv(ruta)

print(df.head())