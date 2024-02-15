# -*- coding: utf-8 -*-
import re

monopthongs = {
    'AO': 'ɔ',
    'AO0': 'ɔ',
    'AO1': 'ɔ',
    'AO2': 'ɔ',
    'AA': 'ɑ',
    'AA0': 'ɑ',
    'AA1': 'ɑ',
    'AA2': 'ɑ',
    'IY': 'i',
    'IY0': 'i',
    'IY1': 'i',
    'IY2': 'i',
    'UW': 'u',
    'UW0': 'u',
    'UW1': 'u',
    'UW2': 'u',
    'EH': 'e',
    'EH0': 'e',
    'EH1': 'e',
    'EH2': 'e',
    'IH': 'ɪ',
    'IH0': 'ɪ',
    'IH1': 'ɪ',
    'IH2': 'ɪ',
    'UH': 'ʊ',
    'UH0': 'ʊ',
    'UH1': 'ʊ',
    'UH2': 'ʊ',
    'AH': 'ʌ',
    'AH0': 'ə',
    'AH1': 'ʌ',
    'AH2': 'ʌ',
    'AE': 'æ',
    'AE0': 'æ',
    'AE1': 'æ',
    'AE2': 'æ',
    'AX': 'ə',
    'AX0': 'ə',
    'AX1': 'ə',
    'AX2': 'ə',
}

dipthongs = {
    'EY': 'eɪ',
    'EY0': 'eɪ',
    'EY1': 'eɪ',
    'EY2': 'eɪ',
    'AY': 'aɪ',
    'AY0': 'aɪ',
    'AY1': 'aɪ',
    'AY2': 'aɪ',
    'OW': 'oʊ',
    'OW0': 'oʊ',
    'OW1': 'oʊ',
    'OW2': 'oʊ',
    'AW': 'aʊ',
    'AW0': 'aʊ',
    'AW1': 'aʊ',
    'AW2': 'aʊ',
    'OY': 'ɔɪ',
    'OY0': 'ɔɪ',
    'OY1': 'ɔɪ',
    'OY2': 'ɔɪ',
}

r_colored_vowels = {
    'ER': 'ɜr',
    'ER0': 'ɜr',
    'ER1': 'ɜr',
    'ER2': 'ɜr',
    'AXR': 'ər',
    'AXR0': 'ər',
    'AXR1': 'ər',
    'AXR2': 'ər',
}

stops = {
    'P': 'p',
    'B': 'b',
    'T': 't',
    'D': 'd',
    'K': 'k',
    'G': 'g',
}

affricates = {
    'CH': 'tʃ',
    'JH': 'dʒ',
}

fricatives = {
    'F': 'f',
    'V': 'v',
    'TH': 'θ',
    'DH': 'ð',
    'S': 's',
    'Z': 'z',
    'SH': 'ʃ',
    'ZH': 'ʒ',
    'HH': 'h',
}

nasals = {
    'M': 'm',
    'EM': 'm̩',
    'N': 'n',
    'EN': 'n̩',
    'NG': 'ŋ',
    'ENG': 'ŋ̍',
}

liquids = {
    'L': 'l',
    'EL': 'ɫ̩',
    'R': 'r',
    'DX': 'ɾ',
    'NX': 'ɾ̃',
}

semivowels = {
    'W': 'w',
    'Y': 'j',
    'Q': 'ʔ'
}

arpa_to_ipa_lookup = {}
arpa_to_ipa_lookup.update(monopthongs)
arpa_to_ipa_lookup.update(dipthongs)
arpa_to_ipa_lookup.update(r_colored_vowels)
arpa_to_ipa_lookup.update(stops)
arpa_to_ipa_lookup.update(affricates)
arpa_to_ipa_lookup.update(fricatives)
arpa_to_ipa_lookup.update(nasals)
arpa_to_ipa_lookup.update(liquids)
arpa_to_ipa_lookup.update(semivowels)


def split_on_capital(camel):
    ' '.join(re.findall('[A-Z][a-z]*', camel.title())).lower()


def arpa_to_ipa(arpa):
    return ' '.join(arpa_to_ipa_lookup[phoneme] for phoneme in arpa.upper().split(' '))
