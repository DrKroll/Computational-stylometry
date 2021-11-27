#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 1.0.0

import sys                                  # Librerías del sistema
import os
import time                                 # Librerías de control
from progress.bar import Bar
import pandas as pd                         # Librerías operativas
import re
from libescansiondos import silabas as slb
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

start_time = time.time()
cuentaritmos = {}
if len(sys.argv) > 1:
    entrada = sys.argv[1]
else:
    entrada = 'prueba.csv'


def screen_clear():
    if os.name == 'posix':
        _ = os.system('clear')
    else:
        _ = os.system('cls')

def renuevalista(slb, esp, n):
    if esp[0] != slb:
        if slb not in [11, 7]:
            if slb != 8:
                esp = [x for x in esp if x != slb]
                esp.insert(3, slb)
            else:
                esp = [8, 11, 8, 11, 8, 7, 6, 10, 12, 5, 9, 14, 4, 13, 3, 15, 2, 3, 4, 1]
        else:
            silva = [slb] + [s for s in [11, 7] if s != slb]
            esp = silva + [s for s in esp if s not in [11, 7]]
        n += 1
    return (esp, n)


def entradadf(indice, texto, p, esperadas, cuentaritmos = {}):
    df.at[indice, 'Personaje'] = p
    if texto == texto:
        tex = re.sub(r'\s{2,}', ' ', texto.strip())
        recuento = slb(texto, esperadas, cuentaritmos)
        nuclei = ''.join(recuento.nucleosilabico)
        sil = int(recuento.ml)
        rim = recuento.rima
        ason = recuento.ason
        rit = recuento.ritmo
        if '[' in texto:
            amb = 3
        else:
            amb = recuento.ambiguo
    else:
        tex = 'X'
        sil = esperadas[0]
        rim = 'X'
        rit = 'X'
        ason = 'X'
        amb = 1
    df.at[indice, 'Texto'] = tex
    df.at[indice, 'Slbs'] = sil
    df.at[indice, 'Ambiguo'] = amb
    df.at[indice, 'NuclVoc'] = nuclei
    df.at[indice, 'Rima'] = ason
    df.at[indice, 'Consonancia'] = rim.lower()
    df.at[indice, 'Ritmo'] = rit

    cuentaritmos = diccionario_ritmos(sil, rit, cuentaritmos)


def diccionario_ritmos(slbs, ritmo, diccionario):
    if slbs not in diccionario.keys():
        diccionario[slbs] = {ritmo: 1}
    elif ritmo not in diccionario[slbs].keys():
        diccionario[slbs][ritmo] = 1
    else:
        diccionario[slbs][ritmo] += 1
    return diccionario


def guarda(dataframe, nombre):
    dataframe['Slbs'] = dataframe['Slbs']
    dataframe['Ambiguo'] = dataframe['Ambiguo']
    dataframe = dataframe.drop_duplicates(subset=['Verso'], keep='last').convert_dtypes()
    dataframe.to_csv(f'{nombre}.out', mode='w', header=True, index=False)


start_time = time.time()                    # Inicia el cronómetro
if len(sys.argv) > 1:                       # Decide qué archivo lee
    entrada = sys.argv[1]
else:
    entrada = 'prueba.csv'

df = pd.read_csv(entrada)                   # Lee el archivo de entrada
df['Jornada'] = df['Jornada']   # Convierte columna de real a int
df['Parlamento'] = df['Parlamento']
df['Verso'] = df['Verso']
nombrebase = entrada.rsplit('.', 1)[0]

anterior = personaje = parlamento = anterior = ''
i = ultima_fila = verso_anterior = 0
bar = Bar('Procesado:', max=len(df.index))
pieza = df.iloc[0]['Pieza']
valores = ([8, 11, 7, 6, 10, 12, 5, 9, 14, 4, 13, 3, 15, 2], 0)

for index, fila in df.iterrows():
    if pieza != fila['Pieza']:
        guarda(df[df['Pieza'] == pieza], pieza)
        pieza = fila['Pieza']
    screen_clear()
    print(f'\nObra:\t\t{fila["Pieza"]}\nJornada:\t{fila["Jornada"]}\n'\
          f'Texto:\t\t{fila["Texto"]}\n')
    bar.next()
    fila = fila.copy()
    if fila['Verso'] > 0:
        if fila['Verso'] > verso_anterior or fila['Parlamento'] == 1:
            vers = fila['Texto']
            pant = fila['Personaje']
            anterior = vers
            verso_anterior = fila['Verso']
            entradadf(index, vers, fila['Personaje'], valores[0], cuentaritmos)
            if index == df.index[-1] or fila['Verso'] < df.at[index+1, 'Verso']:
                valores = renuevalista(int(df.at[index, 'Slbs']),
                                       valores[0],
                                       valores[1])
            esperadas = valores[0]
            cuenta = valores[1]
        else:
            vers = f'{anterior} {fila["Texto"]}'
            personaje = f'{pant} y {fila["Personaje"]}'
            pant = personaje
            anterior = vers
            entradadf(index, vers, personaje, valores[0])
        ultima_fila = index
    else:
        df = df.drop(index)
bar.finish()
guarda(df[df['Pieza'] == pieza], pieza)
print(f'El programa se ejecutó en {time.time() - start_time} segundos')
