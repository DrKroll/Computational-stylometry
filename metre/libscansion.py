# v1.1.0beta1
version = '0.1.1beta1'
import re
import stanza
from fonemas import transcribe
processor_dict = {'tokenize': 'ancora', 'pos': 'ancora',
                  'ner': 'ancora', 'depparse': 'ancora'}
config = {'lang':  'es', 'processors': processor_dict}
nlp = stanza.Pipeline(**config)

habituales = ['juez', 'cruel', 'fiel']
habituales = [[silaba.strip("'").strip() for silaba in palabra] for palabra in
              [transcribe(habitual) for habitual in habituales]]


class lineatexto:
    tonicos = ['yo', 'vos', 'es', 'soy', 'voy', 'sois', 'vais', 'ti']
    detat = ['el', 'la', 'los', 'las', 'mi', 'tu', 'su',
             'me', 'te', 'le', 'nos', 'os', 'les', 'lo', 'se', 'mis', 'tus',
             'sus', 'adonde']
    demostrativos = ['aqueste', 'aquesta', 'aquestos', 'aquestas', 'aquese',
                     'aquesa', 'aquesas', 'aquesos', 'este', 'esta', 'esto',
                     'estos', 'estas', 'ese', 'esos', 'esa', 'esas', 'eso',
                     'aquel', 'aquella', 'aquellos', 'aquellas',
                     'tuyo', 'tuyos', 'tuya', 'tuyas',
                     'suyo', 'suya', 'suyos', 'suyas']
    numeros = ['uno', 'una', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete',
               'ocho', 'nueve']
    cortesia = ['don', 'doña', 'sor', 'fray', 'santo']

    def __init__(self, linea):
        self.linea = linea
        self.__limpio = self.__parchea_linea(self.__preprocesa(self.linea))
        self.palabras = [x['fon'] for x in
                         self.__acento_metrico(
                               self.__separa(self.__limpio))]

    def __preprocesa(self, palabras):
        simbolos = {'(': '.', ')': '.', '—': '. ', '…': '.',  # ',': '. ',
                    ';': '.', ':': '.', '?': '.', '!': '.',
                    'õ': 'o', 'æ': 'ae',
                    'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u'}
        if palabras != palabras:
            palabras = 'ma ma ma ma ma ma mama'
        else:
            for i in simbolos:
                palabras = palabras.replace(i, f'{simbolos[i]} ')
            palabras = re.sub(r'\s*\.+(\w)', r',\1', palabras)
            palabras = re.sub(r'\s*\,+(\w)', r',\1', palabras)
            palabras = re.sub(r'\[|\]|¿|¡|^\s*[\.\,]|»|«|“|”|‘|’|"|-|–',
                           '', palabras)
            palabras = re.sub(r'\s*\.[\.\s]+', ', ', palabras)
            palabras = palabras.strip()
            palabras = palabras[0].upper()+palabras[1:]
        usadas = []
        verso =  nlp(palabras)
        palabras =[]
        for sentence in verso.sentences:
            for word in sentence.words:
                if word.parent.id not in usadas:
                    usadas.append(word.parent.id)
                    word.text = word.parent.text
                    palabras.append(word)
        return palabras

    def __parchea_linea(self, linea):
        palabras = [palabra for palabra in linea
                    if not (palabra.pos == 'PUNCT' and
                            not re.match('\w', palabra.text))
                        and palabra.pos != 'X']
        if len(palabras) > 1:
            if (palabras[-1].parent.text == palabras[-2].parent.text and
                palabras[-1].text != palabras[-2].text):
                palabras[-2].text = palabras[-2].parent.text
                palabras.pop()
        return palabras

    def __separa(self, palabras):
        return [{'text': palabra.text,
                 'pos': palabra.upos,
                 'fon': transcribe(palabra.text),
                 'feats': self.__parse_feats(palabra.feats),
                 'deprel': palabra.deprel,
                 'ton': False} for
                palabra in palabras]

    @staticmethod
    def __marca_acentos_metricos(palabra):
        for idx, silaba in enumerate(palabra['fon']):
            if silaba.startswith("'"):
                if palabra['ton'] is True:
                    silaba = silaba.translate(str.maketrans("aeiou", "AEIOU"))
                palabra['fon'][idx] = silaba.strip("'").strip()
        return palabra

    @staticmethod
    def en_mente(palabra):
        if (len(palabra['text']) > 5 and palabra['text'].endswith('mente') and
            palabra['pos'] == 'ADV'):
            secundario = transcribe(palabra['text'].split('mente')[0]) + (
                ["'men", 'te'])
            palabra['fon'] = secundario
            palabra['tonica'] = True
        return palabra

    def __parchea_palabra(self, palabra):
        if palabra['pos'] == 'PUNCT':
            palabra['pos'] = 'VERB'
        if (palabra['text'].endswith('mente') and len(palabra['text']) > 5
            and palabra['pos'] == 'ADV'):
            palabra = self.en_mente(palabra)
        return(palabra)

    def __acento_metrico(self, palabras):
        postonicos = ['ADV', 'NOUN', 'PROPN',
                      'NUM', 'DET.Dem', 'DET.Int', 'DET.Ind', 'PRON.Com',
                      'PRON.Nom']
        conjat = ['y', 'e', 'ni', 'o', 'u', 'que', 'pero', 'sino', 'mas',
                  'aunque', 'pues', 'porque', 'como', 'pues', 'luego', 'cual',
                  'conque', 'si', 'cuando', 'aunque', 'aun cuando', 'que',
                  'quien', 'donde', 'cuando', 'cuanto', 'como']
        tonicos = ['agora']

        number_flag = False
        noun_flag = False

        for idx, palabra in enumerate(palabras[::-1]):
            palabra = self.__parchea_palabra(palabra)
            ton = palabra['ton']
            txt = palabra['text']
            pos = palabra['pos']
            fts = palabra['feats']
            deprel =  palabra['deprel']
            if idx == 0 or pos in ['ADJ', 'VERB', 'AUX', 'INTJ',
                                   'NOUN', 'PROPN'] or txt in tonicos:
                ton = True
            else:
                if pos in postonicos:
                    ton = True
                if txt in conjat:
                    ton = False
                    if txt.lower() != 'y':
                        number_flag = False
                if pos == ('NUM' and txt not in self.numeros):
                    if number_flag == False:
                        ton = True
                    number_flag = True
                else:
                    number_flag = False

                if pos == 'DET':
                    if fts['PronType'] == 'Poss' or fts['Poss'] == 'Yes':
                        if txt in ['mi', 'tu', 'su', 'mis', 'tus', 'sus']:
                            ton = False
                        elif any([txt.startswith(x)
                                  for x in ['nuestr', 'vuestr']]):
                            if noun_flag == False:
                                ton = True
                        else:
                            ton = True
                    elif fts['Case'] and fts['Case'] in ['Acc,Dat',
                                                         'Dat', 'Acc']:
                        ton = False
                    elif 'Definite' in fts:
                        if fts['Definite'] != 'Def':
                            ton = True
                    else:
                        ton = False
                if pos in ['PROPN', 'NOUN'] or txt in self.cortesia:
                    if noun_flag == True:
                        ton = False
                    else:
                        noun_flag = True
                if pos == 'PRON' and (any([x in txt for x in 'áéíóú'] or
                                      txt in ['yo', 'ella', 'ello', 'ellos',
                                            'ellas', 'nosotros', 'nosotras',
                                            'vosotros', 'vosotras', 'ti']) or
                                      fts['PronType'] in ['Tot', 'Exc', 'Dem'
                                                          'Ind', 'Int',
                                                          'Neg']):
                    ton = True
            palabra['ton'] = ton
            palabras[idx] = self.__marca_acentos_metricos(palabra)
        palabras.reverse()
        return palabras

    @staticmethod
    def __parse_feats(feats):
        dd = {}
        if feats:
            cosas = feats.split("|")
            for i in cosas:
                feature = i.split("=")
                dd[f'{feature[0]}'] = feature[1]
        else:
            dd['None'] = 'None'
        if 'Case' not in dd:
            dd['Case'] = ''
        if 'Poss' not in dd:
            dd['Poss'] = 'No'
        return dd


##############################################################################


class silabas:
    def __init__(self, linea, esperadas, recuentoritmos = {}):
        self.linea = lineatexto(linea).palabras
        self.correcc = self.__corrige_metro(self.linea, esperadas)
        self.silabasmetricas = self.correcc['slbs']
        self.ambiguo = self.correcc['amb']
        self.ason = self.correcc['ason']
        self.rima = self.correcc['cons']
        self.ml = self.correcc['suma']
        #self.rima = self.ultima['consonancia']
        self.nucleosilabico = [item for silaba in
                               [x for y in self.silabasmetricas
                                for x in y]
                               for item in silaba
                               if item in 'AEIOUaeiou']
        self.ritmo = self.__ritmo(self.nucleosilabico)
        print(f'\nVersión: {version}\n')

    @staticmethod
    def sustituye(palabra, reemplazo, texto):
        def func(match):
            g = match.group()
            if g.islower(): reemplazo = reemplazo.lower()
            if g.istitle(): reemplazo = reemplazo.title()
            if g.isupper(): reemplazo = reemplazo.upper()
        return re.sub(palabra, reemplazo, texto, flags=re.I)

    @staticmethod
    def __preferencia_sinalefa(uno, dos, preferencia=0):
        # 2. No juntar sinalefa y diéresis
        # 4. Doble -v v v-
        if uno.lower() == dos.lower():
            preferencia += 1
        if uno.islower() and dos.islower(): # unstressed + unstressed
            preferencia += 4
            if len(dos) > 1:
                if dos[0] in 'àèìòùjw' and all( x in 'aeiou' for                 # Syllabic + semivowel + syllabic
                                                    x in [uno[-1],
                                                          dos[1]]):
                    preferencia -= 8
            elif dos[0] in 'aeoAEO' and uno[-1] in 'iu':                         # high + middle/low
                preferencia += 4
            elif uno[-1] in 'àèìòùjw' and dos[0] in 'aeiou':                     # Semivowel + syllabic
                preferencia += 4
            elif uno[-1] in 'aeiou' and dos[0] in  'àèìòùjw':
                preferencia += 2
            else:
                preferencia += 1
        elif not all(x.isupper() for x in [uno[-1], dos[0]]):
            if uno in 'ieou' and dos[0] == uno.upper():                          # conjunctions y e o u + same sound stressed  
                preferencia = -1
            else:
                    preferencia += 2
        else:
            # 5. Entre tónicas, mejor hiato
            preferencia -= 1
        return preferencia

    def __busca_sinalefas(self, palabras):
        vocales = 'aeioujwiAEIOU'
        semivocales = 'wjàèò'
        # str.maketrans(semivocales, 'iu'))
        sinalefas = []
        preferencia = 0
        for i, palabra in enumerate(palabras):
            if i > 0:
                segunda = palabra[0]
                primera = palabras[i-1][-1]
                posicion = [i-1, len(palabras[i-1]) -1]
                if segunda[0] in semivocales and len(segunda) == 1:
                    if len(palabras) > i:
                        if palabras[i+1][0][0] in vocales:
                            preferencia -= 8
                else:
                    preferencia = 0
                if all([x in vocales for x in [primera[-1], segunda[0]]]):
                    sinalefas.append((posicion,
                                      self.__preferencia_sinalefa(primera,
                                                                  segunda,
                                                                  preferencia))
                                     )
            

        #sinéresis
        for i, palabra in enumerate(palabras):
            if palabra in habituales:
                preferencia = 4
            for j, silaba in enumerate(palabra):
                if j > 0:
                    posicion = [i, j-1]
                    segunda = silaba ####
                    primera = palabra[j-1]
                    if all(x in vocales for x in [primera[-1], segunda[0]]):
                        if primera[-1] == segunda[0]:
                            preferencia = 4
                        sinalefas.append((posicion,
                                        self.__preferencia_sinalefa(primera,
                                                                    segunda) +
                                        preferencia -2))
        sinalefas = sorted(sinalefas, key=lambda tup: (-tup[1], tup[0]))
        return [silaba[0] for silaba in sinalefas]

    @staticmethod
    def __haz_sinalefas(diptongo):
        nosilabicas = {'a': 'à', 'e': 'è', 'i': 'j', 'o': 'ò', 'u': 'w',
                       'A': 'à', 'E': 'è', 'I': 'j', 'O': 'ò', 'U': 'w',
                       'j': 'j', 'w': 'w'}
        vocales = 'aeioujw'
        if diptongo[0][-1] == diptongo[-1][0]:
            diptongo = diptongo[0][:-1] + diptongo[1]
        elif all(x in 'wj' for x in [diptongo[0][-1], diptongo[1][0]]):
            diptongo = diptongo[0] + diptongo[1]
        elif all(x.islower() for x in [diptongo[0][-1], diptongo[1][0]]):
            diptongo = (diptongo[0][:-1] +
                        nosilabicas[diptongo[0][-1]] +
                        diptongo[1])
        elif diptongo[0][-1] in 'IU' and  diptongo[1][0] in 'aeiou':
            diptongo = (diptongo[0][:-1] +
                        nosilabicas[diptongo[0][-1]] +
                        diptongo[1][0].upper())
        else:
            diptongo = (diptongo[0][:-1] +
                        nosilabicas[diptongo[0][-1]] +
                        diptongo[1])
        return diptongo


    @staticmethod
    def __corrige_posicion(lista, posicion, offset):
        for idx, i in enumerate(lista):
            if offset < 0:
                if i[0] == posicion[0]:
                    if posicion[1] >= posicion[1]:
                        lista[idx][1] -= 1
            elif i[0] > posicion[0]:
                if i[0] == posicion[0] + 1:
                    lista[idx][1] += offset
                lista[idx][0] -= 1
            else:
                pass
        return lista


    def __ajusta_silabas(self, palabras, lista_sinalefas):
        if len(lista_sinalefas) < 1:
            return palabras
        else:
            if lista_sinalefas[0][0] < sum([len(palabra)
                                            for palabra in palabras]):
                i_palabra1 = lista_sinalefas[0][0]
                i_silaba1 = lista_sinalefas[0][1]
                palabra = palabras[lista_sinalefas[0][0]]
                l_palabra = len(palabra)

                if i_silaba1 == l_palabra - 1:
                    i_silaba2 = 0
                    i_palabra2 = i_palabra1 + 1
                    if len(palabras[i_palabra1]) > 1:
                        primera = palabras[i_palabra1][:-1]
                        media = [palabras[i_palabra1][-1],
                                 palabras[i_palabra2][0]]
                    else:
                        primera = []
                        media = [palabras[i_palabra1][-1],
                                 palabras[i_palabra2][0]]
                    segunda = palabras[i_palabra2][1:]
                    diptongo = [self.__haz_sinalefas(media)]
                    palabra = primera + diptongo
                    if len(palabras[i_palabra2]) > 1:
                        palabra = palabra + segunda
                    palabras = (palabras[:i_palabra1] +
                                [palabra] + palabras[i_palabra2+1:])
                    lista_sinalefas = self.__corrige_posicion(
                        lista_sinalefas[1:], lista_sinalefas[0], len(primera))
                    # palabras[i_palabra1:i_palabra2] = palabra
                else:
                    i_silaba2 = i_silaba1 + 1
                    primera = palabras[i_palabra1][:i_silaba1]
                    segunda = palabras[i_palabra1][i_silaba2+1:]
                    media = [palabras[i_palabra1][i_silaba1],
                             palabras[i_palabra1][i_silaba2]]
                    diptongo = self.__haz_sinalefas(media)
                    palabra = primera + [diptongo] + segunda
                    palabras[i_palabra1] = palabra
                    lista_sinalefas = self.__corrige_posicion(
                        lista_sinalefas[1:], lista_sinalefas[0], -1)
            return self.__ajusta_silabas(palabras, lista_sinalefas)

    @staticmethod
    def __dipt(diferencia, nucleo):
        ajuste = []
        for i in nucleo:
            if diferencia > 0 and len(i) > 1:
                ajuste += [k for k in i]
            else:
                ajuste += i
        return ajuste

    @staticmethod
    def __busca_hiatos(palabras):
        triptongos = []
        diptongos = []
        for idx, palabra in enumerate(palabras):
            for silaba in palabra:
                diptongo = re.search(
                    r'([aeioujw])([aeioujw])([wj]*)',
                    silaba, re.IGNORECASE)
                if diptongo:
                    diptongos.append(idx)
                    if diptongo.group(3):
                        triptongos.append(idx)
        return diptongos + triptongos

    def __hiato(self, palabras, hiatos, diferencia):
        preferencia = []
        sem2voc = {'j': 'i', 'w': 'u'}
        for idx in hiatos[::-1]:
            if palabras[idx] in habituales:
                preferencia = [idx] + preferencia
            else:
                preferencia += [idx]
        for idx in hiatos[0:diferencia]:
            palabra = palabras[idx]
            for j, silaba in enumerate(palabra):
                diptongo = re.search(r'(\w*[jw])([aeiou]\w*)',
                                     silaba, re.IGNORECASE)
                if diptongo:
                    uno = diptongo.group(1)
                    dos = diptongo.group(2)
                    hiato = [uno.replace(uno[-1], sem2voc[uno[-1]]),
                             dos]
                else:
                    diptongo = re.search(r'(\w*[aeiou])([jw]\w*)',
                                         silaba, re.IGNORECASE)
                    if diptongo:
                        hiato = [uno,
                                 dos.replace(dos[0], sem2voc[dos[0]])]
                        palabra += hiato
            palabras[idx] = palabra
        return palabras

    def __corrige_metro(self, silabas, esperadas):
        ajustadas = silabas
        sinalefas_potenciales = self.__busca_sinalefas(ajustadas)
        hiatos_potenciales = self.__busca_hiatos(ajustadas)
        rima = self.__rima(ajustadas[-1])
        lon_rima = sum([len(palabra)
                        for palabra in ajustadas]) + rima['suma']
        offset = esperadas[0] - lon_rima
        if offset == 0:
            ambiguo = 0
        elif lon_rima - len(sinalefas_potenciales) == esperadas[0]:
            ambiguo = 0
            ajustadas = self.__ajusta_silabas(ajustadas,
                                              sinalefas_potenciales)
        else:
            ambiguo = 1
            if offset < 0:
                if len(sinalefas_potenciales) >= -offset:
                    sinalefas_potenciales = sinalefas_potenciales[0:-offset]
                    ajustadas = self.__ajusta_silabas(ajustadas,
                                                    sinalefas_potenciales)
                else:
                    ajustadas = self.__hiato(ajustadas,
                                             hiatos_potenciales,
                                             offset)
        lon_rima = sum([len(palabra) for palabra in ajustadas]) + rima['suma']
        if lon_rima  != esperadas[0]:
            return self.__corrige_metro(ajustadas, esperadas[1:])
        else:
            return {'slbs': ajustadas, 'amb': ambiguo, 'suma': lon_rima,
                    'ason': rima['asonancia'], 'cons': rima['consonancia']}

    @staticmethod
    def __rima(palabra):
        vocales = 'aeiou'
        suma = {-1: 1, -2: 0, -3: -1}
        for idx, silaba in enumerate(palabra[::-1]):
            if any([x.isupper() for x in silaba]):
                tonica = -idx - 1
                break
        ultimas = palabra[tonica:]
        consonancia = ''.join([x.lower() for x in ultimas])
        asonancia = ''.join([x for x in consonancia if x in vocales])
        return {'tonica': tonica, 'suma': suma[tonica],
                'asonancia': asonancia, 'consonancia': consonancia}

    @staticmethod
    def __ritmo(silabas):
        metro = []
        for silaba in silabas:
            if any(letra.isupper() for letra in silaba):
                metro.append('+')
            else:
                metro.append('-')
        return ''.join(metro)
