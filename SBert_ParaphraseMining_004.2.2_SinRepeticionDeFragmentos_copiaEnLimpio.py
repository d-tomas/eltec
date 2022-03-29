#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Borja Navarro Colorado - Universidad de Alicante
# Artículo "Similitud multilingüe novela europea..."
# 4 de febrero de 2022

#Dadas dos novelas en dos idiomas diferentes, extrae X fragmento al azar de cada una de tamaño Y (sin repetición)
#y calcula su similitud todos contra todos con SenteceBert.
#De la parejas creadas, selecciona solo aquellas que sea entre idiomas diferentes (ignora las comparaciones
#entre fragmento de la misma novela) y
#con una similitud superior a un umbral Z.

#Ejemplo tomado de aquí:
#https://www.sbert.net/examples/applications/paraphrase-mining/README.html

import os
import random

#print('AVISO: este script se ejecuta en el entorno virtual CONDA "huellasEspiritismo".\nPara activarlo:\n\tconda activate huellasEspiritismo\n')


#Recorre el directorio con el corpus y extrae pasajes de una cantidad determinada de palabras.
#El corpus ha sido previamente lematizado.
#Crea un diccionario donde se almacena el nombre de fichero (clave) y el fragmento extraídos (valor).

dir_in = "/home/borja/Documentos/Investigacion/ProyectoNovelaEuropea/corpusELTeC_formato3columnas/CorpusELTeC_Formato3cols/"

configuracion='Aproximación 2 (SentenceBERT), experimento 3: selección aleatoria de pasajes a comparar SIN REPETICIÓN.\n'
configuracion='Tokens (no lemas).\n' #-> ver línea 91

#1. tamaño de la muestra:
tamanno_fragmento = 50
configuracion+="tamaño del fragmento = "+str(tamanno_fragmento)+" palabras.\n"
#2. cantidad de muestras a extraer por novela 
#Cuidado, con tamaño_fragmento 50, no superar 263 muestras, que es el tamaño de la novela más pequeña de SPA4006_Dicenta_LasEsmeraldas.csv
#Aquí hay que controlar tamaño de muestra y cantidad de muestras para que no dé error.
cantidad_muestras = 250 
configuracion+="cantidad de muestras por novela = "+str(cantidad_muestras)+".\n"
# 3. umbral de similitud.
umbral_similitud = 0.50
configuracion+="Umbral de similitud = "+str(umbral_similitud)+".\n"

print('Tamaño de los fragmentos:', tamanno_fragmento,'palabras.')
print('Tamaño máximo permitido:',tamanno_maximo_permitido,'palabras.')
print('Cantidad de fragmentos por novela:', cantidad_muestras,'muestras.')
print('Umbral de similitud:', umbral_similitud)


def separa_lista(lst, n):  
    for i in range(0, len(lst), n): 
        yield lst[i:i + n]

def extrae_fragmentos_aleatorio(lst, cantidad_elementos, tamanno):
    salida = []
    salida = random.sample(lst,cantidad_elementos)
    return salida

print('Cargando el corpus...')
#El siguiente bucle extrae los fragmentos y crea un dicciario
# "textos" donde cada pasaje tiene un ID.
textos = {}
for base, directorios, ficheros in os.walk(dir_in):
    for fichero in ficheros:
        ficheroEntrada = base+"/"+fichero
        print('Procesando', fichero)
        with open(ficheroEntrada, 'r') as f:
            
            novela_analizada = f.read()
            novela_lemas = [] #Aquí se almacenan todos los lemas.
            tokens = novela_analizada.split('\n')
            for token in tokens:
                analisis = token.split()
                if len(analisis) > 2:
                    lema = analisis[0] #Aquí se puede poner el filtro de signos de puntuación, palabras funcionales, etc.
                    novela_lemas.append(lema)

            fragmentos = separa_lista(novela_lemas, tamanno_fragmento) #Separa al lista con todos los lemas en grupos de "tamanno_fragmento"
            fragmentos_final_TOTAL = list(fragmentos)
            print('Cantidad TOTAL de fragmentos en la novela', len(fragmentos_final_TOTAL))
            fragmentos_final=[]
            for i in fragmentos_final_TOTAL:#fragmentos:
                if len(i) == tamanno_fragmento:
                    fragmentos_final.append(i)
            
            #print('Cantidad de palabras:', len(novela_lemas))
            print('Cantidad de fragmentos con el mismo tamaño', len(fragmentos_final))
            #print('en grupos de', tamanno_fragmento)
            #print('Comprobador, se esperan estos fragmentos:', str(len(novela_lemas)/tamanno_fragmento) )
            fragmento_final_aleatorio = extrae_fragmentos_aleatorio(fragmentos_final, cantidad_muestras, tamanno_fragmento)
            #print(fragmento_final_aleatorio[0], len(fragmento_final_aleatorio[0]))
            #for item in fragmento_final_aleatorio:
            #    if len(item) != tamanno_fragmento:
            #        print('Error. Fragmento de tamaño incorrecto:', len(item))
            #print(fragmento_final_aleatorio[len(fragmento_final_aleatorio)-1],len(fragmento_final_aleatorio[len(fragmento_final_aleatorio)-1]) )
            #print('Cantidad de fragmentos aleatorios:', len(fragmento_final_aleatorio)) #->Comprobar que la cantidad es la correcta.

            i=0
            for fragmento in fragmento_final_aleatorio:
                identificador = fichero+'#'+str(i) #Creamos un ID único para cada fragmento, que es el nombre del fichero más un número.
                pasaje_cadena = ''
                for token in fragmento:
                    pasaje_cadena+=token+' '
                textos[identificador] = pasaje_cadena
                i+=1
            print('Cantidad de fragmentos por novela', i) #--> Cantidad de fragmentos insertado en el diccionario. Debe ser igual a len(fragmentos_final)
            print('Cantidad total de fragmentos extraídos', len(textos)) #--> Cantidad de elementos del diccionario. Se van sumando con forme se procesa cada novela.
            #for item in textos.items():
            #    print(item)
        
print('\n######################\n')

lista_completa_pasajes = [] #Lista donde se almacenan todos los pasajes del diccionario "textos". Es la entrada de SBert.
for pasaje in textos.values():
    lista_completa_pasajes.append(pasaje)
print(len(lista_completa_pasajes), '-> lista completa de pasajes')

print('Cargando SentenceBERT...')
from sentence_transformers import SentenceTransformer, util

print('Cargando modelo...')
#model = SentenceTransformer('all-MiniLM-L6-v2') # Modelo del ejemplo. Solo inglés.
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') # Modelo multilingüe.

print('Calculando similitudes...')
paraphrases = util.paraphrase_mining(model, lista_completa_pasajes, show_progress_bar=True)

print('\n######################\n')

#Procesa todas las similitudes calculadas.
#Si son de idiomas diferentes superior a umbral, las almacena en el diccionario "resultado".
#Formato {similitud=[(ID1, ID2, (fragmento1, fragmento2))]}
#Si son del mismo idioma, en el dicc. "resultado_monolingue"

print('Generando resultado...')
print('Cantidad de enparejamientos:', len(paraphrases))
print('\tExtrayendo parejas multilingües con similitud superior a', umbral_similitud)
resultado = {}
resultado_mololingue = {}
numerito=1
for paraphrase in paraphrases:
    score, i, j = paraphrase
    if score > umbral_similitud:
        print('Ten paciencia. Esto tarda. Llevo procesados', numerito, 'fragmentos.')
        numerito+=1
    
        pasaje1 = lista_completa_pasajes[i]
        pasaje2 = lista_completa_pasajes[j]
        fichero1 = list(textos.items())[i][0]
        fichero2 = list(textos.items())[j][0]
        idioma1 = fichero1[:3]
        idioma2 = fichero2[:3]
        if idioma1 != idioma2:
            textos_similares = [pasaje1, pasaje2]
            resultado[score] = [fichero1, fichero2, textos_similares]
        elif fichero1.split('#')[0] != fichero2.split('#')[0]: #Se les quita el índice y queda solo el nombre del fichero.
            textos_similares = [pasaje1, pasaje2]
            resultado_mololingue[score] = [fichero1, fichero2, textos_similares]
#print(resultado)

print('\tOrdenando resultados...')
ordenado = sorted(resultado.items(), reverse=True)
ordenado_monolingue = sorted(resultado_mololingue.items(), reverse=True)

print('\tGenerando fichero final...')
salida = 'Similitud\tfichero1\tfichero2\ttexto1\ttexto2\n'
for item in ordenado:
    #print(item)
    similitud = item[0]
    fichero1=item[1][0]
    fichero2=item[1][1]
    fragmento1 = item[1][2][0]
    fragmento2 = item[1][2][1]
    
    salida+=str(similitud)+'\t'+fichero1+'\t'+fichero2+'\t'+fragmento1+'\t'+fragmento2+'\n'
#print(salida)

salida_monolingue = 'Similitud\tfichero1\tfichero2\ttexto1\ttexto2\n'
for item in ordenado_monolingue:
    #print(item)
    similitud = item[0]
    fichero1=item[1][0]
    fichero2=item[1][1]
    fragmento1 = item[1][2][0]
    fragmento2 = item[1][2][1]
    
    salida_monolingue+=str(similitud)+'\t'+fichero1+'\t'+fichero2+'\t'+fragmento1+'\t'+fragmento2+'\n'

out = open("SBert_ParaphraseMining004_ModeloAleatorioConCorpusControl_Resultados11_bilingue.csv", 'w')
out.write(salida)
out.close()

out = open("SBert_ParaphraseMining004_ModeloAleatorioConCorpusControl_Resultados11_monolingue.csv", 'w')
out.write(salida_monolingue)
out.close()

out_configuracion = open("SBert_ParaphraseMining004_ModeloAleatorioConCorpusControl_Resultados11_configuracion.csv", 'w')
out_configuracion.write(configuracion)
out_configuracion.close()

print('Hecho')
