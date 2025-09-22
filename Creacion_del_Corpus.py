from PyPDF2 import PdfReader
import re
import numpy as np

def pdf_a_txt(ruta, indice_inicial, indice_final = 0):

  ruta_txt = ruta.replace('.pdf', '.txt')

  with open(ruta, 'rb') as pdf:
    pdf = PdfReader(pdf)
    texto = ""
    if indice_final == 0:
      cantidad_de_paginas = len(pdf.pages)
    else:
      cantidad_de_paginas = indice_final
    for num_pagina in range(indice_inicial, cantidad_de_paginas):
      pagina = pdf.pages[num_pagina]
      texto += pagina.extract_text()

  with open(ruta_txt, 'w') as txt:
    txt.write(texto)


def definir_diccionario(txt_a_separar, ruta_txt_separado):

  palabras_unicas = []

  with open(txt_a_separar, 'r', encoding = 'utf-8') as txt:
    contenido_txt = txt.read() #bytes
    palabras = re.findall(r'[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+|[.,]', contenido_txt)
    for palabra in palabras:
      palabra = palabra.lower()
      if palabra not in palabras_unicas:
        palabras_unicas.append(palabra)

  with open(ruta_txt_separado, 'w') as txt_unicas:
    palabras_unicas = str(palabras_unicas)
    txt_unicas.write(palabras_unicas)
    
def crear_corpus(lista_txts, ruta_corpus):
  with open(ruta_corpus, 'w', encoding = 'utf-8') as corpus:
    for txt in lista_txts:
      with open(txt, 'r', encoding = 'utf-8') as txt:
        corpus.write(txt.read() + '\n')
  return corpus

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
      palabra = palabra.lower()
      if palabra in V:
        corpus_indices.append(V[palabra])
  return corpus_indices






