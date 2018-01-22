# -*- coding: utf-8 -*-

import codecs

import re

english_punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
s = '！……（）「」【】、：；""\'《》？／。，“”・―〈〉〞'
chinese_punctuations = s.decode('utf-8')

sp = ['<S>','<UNK>','<EOS>']
def is_chinese_string(sString):
    for c in sString:
        if c >= u'\u4e00' and c <= u'\u9fa5':
            continue
        else:
            return False
    return True

def is_english(s):
    try:
        if 64 < ord(s) < 91 or 96 < ord(s) < 123:
            return True
        return False
    except Exception as e:
        return False


def is_char(sString):
    len(sString)

with codecs.open('/Users/ruiyangwang/dict.txt', 'r', 'utf-8') as f:
    with codecs.open('/Users/ruiyangwang/dict_clean.txt', 'w', 'utf-8') as w:
        stat = 0
        for line in f:
            temp = line.strip().split()
            if is_chinese_string(temp[0]) or temp[0] in english_punctuations or temp[0] in sp or temp[0] in chinese_punctuations:
                w.write(line)
            else:
                stat += int(temp[1])

print stat



