#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 1.0.0

import sys                                  # Librerías del sistema
import os
import time                                 # Librerías de control
from progress.bar import Bar
import pandas as pd                         # Librerías operativas
import re
from libscansion import silabas as slbs
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

start_time = time.time()
count_rythms = {}
if len(sys.argv) > 1:
    input_file = sys.argv[1]
else:
    input_file = 'test.csv'


def screen_clear():
    if os.name == 'posix':
        _ = os.system('clear')
    else:
        _ = os.system('cls')

def refresh_list(syllables, expected, n):
    if expected[0] != syllables:
        if syllables not in [11, 7]:
            if syllables != 8:
                expected = [x for x in expected if x != syllables]
                expected.insert(3, syllables)
            else:
                expected = [8, 11, 7, 6, 10, 12, 5, 9, 14, 4, 13, 3, 15, 4]
        else:
            sylva = [syllables] + [s for s in [11, 7] if s != syllables]
            expected = sylva + [s for s in expected if s not in [11, 7]]
        n += 1
    return (expected, n)


def update_dataframe(idx, text, p, expected, count_rythms = {}):
    df.at[idx, 'Character'] = p
    if text == text:
        tex = re.sub(r'\s{2,}', ' ', text.strip())
        grandtotal = slbs(text, expected, count_rythms)
        nuclei = ''.join(grandtotal.nucleosilabico)
        syl = int(grandtotal.ml)
        rim = grandtotal.rima
        aso = grandtotal.ason
        rhy = grandtotal.ritmo
        if '[' in text:
            amb = 3
        else:
            amb = grandtotal.ambiguo
    else:
        tex = 'X'
        syl = expected[0]
        rim = 'X'
        rhy = 'X'
        aso = 'X'
        amb = 1
    df.at[idx, 'Text'] = tex
    df.at[idx, 'Syllables'] = syl
    df.at[idx, 'Ambiguous'] = amb
    df.at[idx, 'Nuclei'] = nuclei
    df.at[idx, 'Assonance'] = aso
    df.at[idx, 'Consonance'] = rim.lower()
    df.at[idx, 'Rtyhtm'] = rhy

    count_rythms = rhythms_dictionary(syl, rhy, count_rythms)


def rhythms_dictionary(slb, rhy, dic):
    if slb not in dic.keys():
        dic[slb] = {rhy: 1}
    elif rhy not in dic[slb].keys():
        dic[slb][rhy] = 1
    else:
        dic[slb][rhy] += 1
    return dic


def store(dataframe, name):
    dataframe['Syllables'] = dataframe['Syllables']
    dataframe['Ambiguous'] = dataframe['Ambiguous']
    dataframe = dataframe.drop_duplicates(subset=['Verse'],
                                          keep='last').convert_dtypes()
    dataframe.to_csv(f'{name}.out', mode='w', header=True, index=False)


start_time = time.time()                    # Start counting time
df = pd.read_csv(input_file)
df['Act'] = df['Act']                       # real to int
df['Speech'] = df['Speech']
df['Verse'] = df['Verse']
basename = input_file.rsplit('.', 1)[0]
previous = character = parlamento = previous = ''
i = last_row = previous_verse = 0

bar = Bar('Processed:', max=len(df.index))
title = df.iloc[0]['Title']
values = ([8, 11, 7, 6, 10, 12, 5, 9, 14, 4, 13, 3, 15, 2], 0)

for idx, row in df.iterrows():
    if title != row['Title']:
        store(df[df['Title'] == title], title)
        title = row['Title']
    screen_clear()
    print(f'\nTitle:\t\t{row["Title"]}\nAct:\t{row["Act"]}\n'\
          f'Text:\t\t{row["Text"]}\n')
    bar.next()
    row = row.copy()
    if row['Verse'] > 0:
        if row['Verse'] > previous_verse or row['Speech'] == 1:
            vers = row['Text']
            previous_character = row['Character']
            previous = vers
            previous_verse = row['Verse']
            update_dataframe(idx, vers, row['Character'], values[0],
                             count_rythms)
            if idx == df.index[-1] or row['Verse'] < df.at[idx+1, 'Verse']:
                values = refresh_list(int(df.at[idx, 'Syllables']),
                                       values[0],
                                       values[1])
            expected = values[0]
            count = values[1]
        else:
            vers = f'{previous} {row["Text"]}'
            character = f'{previous_character} y {row["Character"]}'
            previous_character = character
            previous = vers
            update_dataframe(idx, vers, character, values[0])
        last_row = idx
    else:
        df = df.drop(idx)
bar.finish()
store(df[df['Title'] == title], title)
print(f'The programme finished in {time.time() - start_time} seconds')
