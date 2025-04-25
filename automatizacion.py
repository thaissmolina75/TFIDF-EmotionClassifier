#Cargar librerias
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


nltk.download('stopwords')

#Limpieza de texto
def limpiarTexto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(r"[^a-z\s]", "", texto)
    palabras = texto.split()
    palabras = [p for p in palabras if p not in stopwords.words("english")]
    return " ".join(palabras)
#limpiar cada frase

vectorizador = TfidfVectorizer()

dataset = load_dataset("dair-ai/emotion")

df = dataset["train"].to_pandas()   
df["textoLimpio"] = df["text"].apply(limpiarTexto)
X = vectorizador.fit_transform(df["textoLimpio"])
y = df["label"]

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_entrenamiento, y_entrenamiento)


#Ejemplos
y_pred = modelo.predict(X_prueba)
print(classification_report(y_prueba, y_pred))
