from Predecir_cbow import predecir_corpus_completo
from Funciones_auxiliares import crear_tokens, convertir_corpus
import numpy as np

def obtener_accuracy(predicciones, top_k):
    aciertos_estrictos = 0
    aciertos_flexibles = 0
    total = len(predicciones)
    for _, real, predicho in predicciones:
        if real == predicho:
            aciertos_estrictos += 1
        if real in top_k:
            aciertos_flexibles += 1
    estricto = round((aciertos_estrictos / total), 2)
    flexible = round((aciertos_flexibles / total), 2)
    
    return estricto, flexible

        
        
        