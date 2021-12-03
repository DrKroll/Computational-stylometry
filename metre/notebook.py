#!/usr/bin/env python
# coding: utf-8
# FIRST PART
# 1. The word: phonological segments
# 2. The verse: prosody
# 3. The text: metatextual structure

from libscansion import transcribe, nlp, silabas as slbs
from silabeador import silabea
from re import sub

# A verse
# Let's make a list of it
text = 'Sueña el rico en su riqueza,'.split()

text

# Hyphenation (orthographic spelling)
silabea('Sueña')

syllables = []
[syllables.append(silabea(word)) for word in text]
syllables

# Phonological transcription (almost)
transcribe('Sueña')

phonemes = []
[phonemes.append(
    transcribe(sub(r'[^\w\s]','',word)))
               for word in text]
phonemes

for word in text:
    print(word)
    w = sub(r"[^\w\s]","",word)
    print(f'"{w}"')
    print(transcribe(w))

morphosyntax = nlp('Sueña el rico en su riqueza')
morphosyntax

prosody = []
for idx, word in enumerate(phonemes):
    stressed = False
    if morphosyntax.sentences[0].words[idx].upos in ['VERB', 'NOUN', 'ADJ']:
        stressed = True
    prosody.append((word, stressed))
prosody

sentence = ''
for word in prosody:
    sentence += ' '+' '.join([syllable if word[1] else syllable.strip("'")
                              for syllable in word[0]])
print(sentence)

# Probable lenghts
length = [8, 11, 77]

# Verse parsing
verso = slbs('Sueña el rico en su riqueza', length)

# Prosodic stress
print(verso.silabasmetricas)
print(verso.ritmo)
print(verso.rima)
print(verso.ason)
print(verso.ml)
print(verso.nucleosilabico)
print(verso.ambiguo)

########################################

# SECOND PART --+---+-

########################################3
# 1. Modules
# 2. Optional visualisation variables
# 3. Preprocessing functions
# 4. Tabular functions
# 5. Statistic functions
# 6. Body

# 1. MODULES

# Import modules

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import collections
import re
from statsmodels.multivariate.manova import MANOVA

# 2. VISUALISATION VARIABLES
# Number of lines shown &c.

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# 4. TABULAR FUNCTIONS

def samples(data, authors, nsamples):
    import numpy as np
    sample = pd.DataFrame()
    for i in authors:
        g = data.loc[data['Author'] == i].groupby('Title')
        a = np.arange(g.ngroups)
        np.random.shuffle(a)
        sample = sample.append(data.loc[df['Author']
                                        == i][g.ngroup().isin(a[:nsamples])])
    return sample

def recount(data, columna):
    dfritmo = pd.DataFrame()
    valores = pd.unique(data[columna])
    obras = pd.unique(data['Title'])
    for obra in obras:
        nversos = len(data[data['Title'] == obra])
        cuentas = {}
        autor =  data.loc[data['Title'] == obra][
            'Author'].value_counts()[:1].index.tolist()[0]
        for valor in valores:
            fila = {'Title': obra, 'Author': autor}
            cuenta = len(data.loc[data['Title'] == obra][data[columna]
                                                         == valor])
            relcuenta = cuenta/nversos
            fila = {'Title': obra, 'Author': autor, columna: valor,
                    'Count': cuenta, 'RelCount' : relcuenta}
            if cuenta > 0:           
                dfritmo = dfritmo.append(fila, ignore_index=True)
    return dfritmo.convert_dtypes()

def longformat(data, columna, cuenta='Count'):
    dflargo = pd.DataFrame()
    obras = pd.unique(data['Title'])
    variables = pd.unique(data[columna])
    calderon = lope = mira = 0
    for obra in obras:
        subconjunto = data[data['Title'] == obra]
        autor = max(subconjunto['Author'])
        fila = {'Author': autor, 'Title': obra}
        for var in variables:
            suma = subconjunto[subconjunto[columna] == var]['Count'].sum()
            fila.update({var: suma})
        dflargo = dflargo.append(fila, ignore_index=True)
    return dflargo.convert_dtypes()

def min_freq(data, columna, minimo):
    return data.groupby(columna).filter(lambda x : (
        x[columna].count()>=minimo).any()).convert_dtypes()

#5. STATISTIC FUNCTIONS

def analyse(data, j_test=pd.DataFrame(), analisis='k', test=0.2, alea = 42,
            plot='rbf', n=3):
    X = data.drop('Author', axis=1).values
    y = data['Author'].values
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test,
                                                        random_state=alea)
    if analisis == 'k':
        from sklearn.neighbors import KNeighborsClassifier
        neighbors = np.arange(1,12)
        train_accuracy =np.empty(len(neighbors))
        test_accuracy = np.empty(len(neighbors))    
        for i,k in enumerate(neighbors):
            modelo = KNeighborsClassifier(n_neighbors=k)
            modelo.fit(X_train,y_train)
            train_accuracy[i] = modelo.score(X_train,y_train)
            test_accuracy[i] = modelo.score(X_test, y_test)
        if plot == 'y':
            plt.rcParams["figure.figsize"] = (10,8)
            plt.title('k-NN number of neighbours')
            plt.plot(neighbors, test_accuracy, label='Test Accuracy')
            plt.plot(neighbors, train_accuracy, label='Training accuracy')
            plt.legend()
            plt.xlabel('Number of neighbours')
            plt.ylabel('Accuracy')
            plt.show()
        else:
            print(test_accuracy)
        modelo = KNeighborsClassifier(n_neighbors=n)
        modelo.fit(X_train,y_train)
    elif analisis == 'r':
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    
        j_test = sc.transform(j_test)
        from sklearn.linear_model import LogisticRegression
        modelo = LogisticRegression(max_iter=1000,  random_state=alea)
        modelo.fit(X_train,y_train)
        y_pred = modelo.predict(X_test)
    elif analisis == 's':
        from sklearn import svm
        modelo = svm.SVC(kernel=plot)
        modelo.fit(X_train,y_train)
    else:
        return False
    from sklearn.metrics import classification_report
    print(f'Score: {modelo.score(X_test,y_test)}')
    from sklearn.metrics import confusion_matrix
    y_pred = modelo.predict(X_test)
    print(modelo.predict(j_test))
    return modelo

#6. IT STARTS HERE

# Load input file
entrada = 'data.csv'
df = pd.read_csv(entrada)
df['Rhythm'] = df['Rhythm'].str.replace('+','X').str.replace('-','o')
df.head()

# Show authors
authors = df['Author'].unique()
print(authors)

# Show titles
titles = df['Title'].unique()
len(titles)

df.groupby(['Author','Title']).size()

# Yo make things simpler later
all_authors = ['Calderón', 'Lope', 'Mira', 'X']
candidates = ['Calderón', 'Lope', 'Mira']
disputed = 'X'

# I just want octosyllabic verses
df = df.loc[df['Syllables'] == 8]

# Candidates
candidatesdf = df[df['Author'] != disputed]

# Disputed
disputeddf = df.loc[df['Author'] == disputed]

# I don't want very unusual hythms 
df = min_freq(df, 'Rhythm',  100)

disputeddf = df.loc[df['Author'] == disputed]
candidatesdf = df.loc[df['Author'] != disputed]

# Count rhythms
dfcount = recount(df, 'Rhythm')
dfcandidatescount = recount(candidatesdf, 'Rhythm')

# Hustogram
plt.figure(figsize=(16,8))
plt.xticks(rotation=45)
plot = sns.histplot(x='Rhythm', hue='Author', data=candidatesdf,
                    multiple="stack") #, palette=paleta,multiple='stack'

# Variance (boxes)
plt.figure(figsize=(16,8))
plt.xticks(rotation=45)
plot = sns.boxplot(x='Rhythm', y='Count', hue='Author', data=dfcandidatescount,
                   medianprops=dict(color="white", alpha=0.7))

# Long format (Eacxh rhtyhm is a column)
dflong = longformat(dfcount, 'Rhythm', 'Count')
dfcandidateslong = longformat(dfcandidatescount, 'Rhythm', 'Count')

# Scatterplot (dots) --+---+-
plt.xticks(rotation=45)
plot = sns.scatterplot(data=dfcandidateslong, x='oXooXoXo', y='ooXoooXo',
                       hue="Author")
#    ooXoooXo oooXooXo oXooXoX oooXooXo oXooXoXo

# Logistic regresion k-NN & SVN
# PArameters sampling
testr = 0.01
testk = 0.2
tests = 0.01
# Randomisation factor (the answer to life, the universe and everything)
alea = 42
# Number of neighbours
kn = 4
# Just the required columns
data = dflong.drop('Title', axis=1)


#tabla, j_test=pd.DataFrame(), test=0.2, alea = 42, plot='n', analisis='k', n=12
print('\n\n\n******************************************\n\Title:\tX:'\
      f'{dflong.loc[dflong["Author"] == "X"]["Title"].max()}\n')
trainer = data.loc[data['Author'].isin(candidates)]
to_test = data.loc[data['Author'] == 'X'].drop('Author', axis=1).values
    
print('Regresion')
analyse(trainer, to_test, 'r', testr, alea,'n', 3)
  
print('\n-------------------------------------------------------\n\nSVN')
#testY = dfreglargo[dfreglargo['Autor'] == autor].drop('Autor', axis=1).values
analyse(trainer, to_test,  's', tests, alea, 'rbf')
    
print('\n-------------------------------------------------------\n\nKnn:\n')
analyse(trainer, to_test,  'k', testk, alea, 'y', kn)

