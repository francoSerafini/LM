from Funciones_auxiliares import aplicar_softmax, obtener_clave
import numpy as np

def predecir_cbow(W, W_s, contexto, diccionario):
  suma_embeddings = 0
  C = len(contexto)
  for palabra in contexto:
    indice_palabra = int(diccionario[palabra])
    suma_embeddings += W[diccionario[palabra]]
  h =  (1/C) * suma_embeddings
  u = W_s.T @ h
  y = aplicar_softmax(u)
  prediccion = np.argmax(y)
  return obtener_clave(diccionario, prediccion)