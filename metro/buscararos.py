#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from sys import argv

entrada = argv[1]

df = pd.read_csv(entrada)

lista = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

raros = df.loc[df['Slbs'].isin(lista)]

raros.to_csv(f'raros.csv', index=False)