from Funciones_auxiliares import aplicar_softmax, obtener_clave
import numpy as np

def predecir_cbow(W, W_s, contexto, diccionario, k=2):
  suma_embeddings = 0
  C = len(contexto)
  for palabra in contexto:
    suma_embeddings += W[diccionario[palabra]]
  h =  (1/C) * suma_embeddings
  u = W_s.T @ h
  y = aplicar_softmax(u)
  prediccion = np.argmax(y)
  top_k = np.argsort(-y)[0:k]
  print(f'Palabra predicha: {obtener_clave(diccionario, prediccion)}')
  return prediccion, top_k

def predecir_corpus_completo(W, W_s, cs_pos, diccionario):
  predicciones = []
  for c_po in cs_pos:
    prediccion = predecir_cbow(W, W_s, c_po[0], diccionario)
    predicciones.append([c_po[0], c_po[1], prediccion])
  return predicciones


    