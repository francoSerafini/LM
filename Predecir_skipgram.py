import numpy as np
from Funciones_auxiliares import aplicar_softmax, obtener_clave


def predecir_skipgram(W, W_s, palabra_central, diccionario):
  i_palabra_central = diccionario[palabra_central]
  h =  W[i_palabra_central]
  u = W_s.T @ h
  y = aplicar_softmax(u)
  y_mayores = np.argsort(-y)[0:4]
  prediccion = [obtener_clave(diccionario, y_mayores[0]), obtener_clave(diccionario, y_mayores[1]), obtener_clave(diccionario, y_mayores[2]), obtener_clave(diccionario, y_mayores[3])]
  return prediccion 