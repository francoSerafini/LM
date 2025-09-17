import numpy as np
import re


def generar_contextos(corpus, C):
  pares = []
  indices = []
  indice = 0

  while len(indices) != C:
    if indice != C/2:
      indices.append(indice)
    indice += 1 #[0,1,3,4]

  for i in range(C//2, len(corpus) - C//2):
    contexto = []
    palabra_central = corpus[i]

    for j in range(len(indices)):
      contexto.append(corpus[indices[j]])
      indices[j] += 1
    pares.append([contexto, palabra_central])

  return pares

def obtener_clave(diccionario, valor):
  for c, v in diccionario.items():
    if diccionario[c] == valor:
      print(c)
      return c
  
def calcular_exitacion_e_o(c_po, W, C):
  suma_Vps = 0
  for palabra in c_po[0]:
    suma_Vps += W[int(palabra)]
  return (1/C) * suma_Vps

def aplicar_softmax(u):
  u_max = np.max(u)  # estabiliza restando el máximo
  exp_u = np.exp(u - u_max)
  return exp_u / np.sum(exp_u)

def generar_one_hot(c_po, V_cardinal):
  one_hot = np.zeros(V_cardinal)
  one_hot[c_po[1]] = 1
  return one_hot

def actualizar_W(W, c_po, EH, tasa_aprendizaje, C):
  for palabra in c_po[0]:
    W[int(palabra)] = W[int(palabra)] - (tasa_aprendizaje * 1/C * EH) #1XN - 1xN
  return W

def crear_tokens(diccionario):
  tokens = {}
  with open(diccionario, 'r', encoding = 'utf-8') as dicc:
    palabras = dicc.read()
    palabras = palabras.strip('[]')
    palabras = palabras.split(',')
    for i in range(len(palabras)):
      if palabras[i] == "'":
        tokens[','] = i
      else:
        tokens[palabras[i].strip().strip("'")] = i
  return tokens

def convertir_corpus(V, corpus):
  corpus_indices = []
  with open(corpus, 'r', encoding = 'utf-8') as corpus:
    palabras = corpus.read()
    palabras = re.findall(r'[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+|[.,]', palabras)
    for palabra in palabras:
      if palabra in V:
        corpus_indices.append(V[palabra])
  return corpus_indices



