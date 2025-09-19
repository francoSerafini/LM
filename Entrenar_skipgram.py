import numpy as np
from Funciones_auxiliares import aplicar_softmax, generar_one_hot, 

def entrenar_skipgram(V, cota, N, C, corpus, tasa_aprendizaje = 0.1):

  ciclos = 0
  V_cardinal = len(V)
  W = np.random.normal(0, 0.1, size=(V_cardinal, N))
  W_s = np.random.normal(0, 0.1, size=(N, V_cardinal))
  Cs_POs = generar_contextos_skipgram(corpus, C) #lista con contextos y palabras objetivos

  #while error > 0 and ciclos < cota:
  while ciclos < cota:

    for c_po in Cs_POs:

      h = W[c_po[0]] #NX1
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
      W[c_po[0]] = W[c_po[0]] - (tasa_aprendizaje * EH)

    ciclos += 1

  return W, W_s