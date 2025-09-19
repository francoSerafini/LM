import numpy as np
from Funciones_auxiliares import generar_contextos, calcular_exitacion_e_o, aplicar_softmax, generar_one_hot, actualizar_W, crear_tokens, convertir_corpus 



def entrenar_cbow(V, ciclos, N, C, corpus, tasa_aprendizaje = 0.1, W = None, W_s = None):

  ciclo = 0
  V_cardinal = len(V)
  p = 0
  Cs_POs = generar_contextos(corpus, C) #lista con contextos y palabras objetivos
  
  if W == None:
    W = np.random.normal(0, 0.1, size=(V_cardinal, N))
    W_s = np.random.normal(0, 0.1, size=(N, V_cardinal))

  while ciclo < ciclos:
    
    for c_po in Cs_POs:
      
      p += 1
      print(f'Palabra nro {p} de {len(corpus)}, del ciclo nro {ciclo}')

      h = calcular_exitacion_e_o(c_po, W, C) #NX1
      #print('h', h.shape)
      u = W_s.T @ h #V_card x 1
      #print('u', u.shape)
      y_j = aplicar_softmax(u) #card_v x 1
      #print('y_j', y_j.shape)
      t_j = generar_one_hot(c_po, V_cardinal) #Card_vx1
      #print('t_j', t_j.shape)
      e_j = y_j - t_j #Card_v x 1
      #print('e_j', e_j.shape)
      W_s = W_s - tasa_aprendizaje * np.outer(h, e_j) #Nxv_card
      EH = W_s @ e_j #NxV @ Vx1 ---> NX1
      W = actualizar_W(W, c_po, EH, tasa_aprendizaje, C)

    ciclo += 1
    p = 0
    np.savez(f"./pesos{N}{ciclo}.npz", W = W, W_s = W_s)

  return W, W_s

diccionario = './TXTS/Diccionario.txt'
corpus = './TXTS/Corpus.txt'

V = crear_tokens(diccionario)
V_cardinal = len(V)
corpus_i = convertir_corpus(V, corpus)
print(f'La cantidad de palabras en el diccionario es {V_cardinal}, el tamalo del corpus es {len(corpus_i)}')

W, W_s = entrenar_cbow(V, 30, 100, 4, corpus_i)