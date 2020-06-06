#!/usr/bin/env python3


import os
import sys
import json
import argparse


VERBOSE = 0


def read_bab2min(fname):
    '''
    $ head -3 raw/bab2min.txt
    이  VCP 0.018279601
    있  VA  0.011699048
    하  VV  0.009773658
    '''
    stopwords = []
    with open(fname) as fp:
        for buf in fp:
            line = buf.rstrip().split('\t')
            if len(line) != 3:
                continue
            morph, tag, ratio = line
            stopword = '{}/{}'.format(morph, tag)
            stopwords.append(stopword)
    return stopwords


def read_6(fname):
    '''
    $ head -3 raw/6.txt
    ["!","\"","$","%","&","'","(",")","*","+",",","-", ...]
    '''
    stopwords = []
    with open(fname) as fp:
        line = fp.read().rstrip()
        stopwords = json.loads(line)
    return stopwords


def read_lined_stopwords(fname):
    '''
    $ head -3 raw/ranksnl.txt
    아
    휴
    아이구
	...

    $ head -3 raw/spikeekips.txt
    가
    가까스로
    가령
	...
    '''
    stopwords = []
    with open(fname) as fp:
        for buf in fp:
            line = buf.rstrip()
            word = line
            stopwords.append(word)
    return stopwords


def read_stopwords_iso(fname):
    stopwords = []
    with open(fname) as fp:
        line = fp.read().rstrip()
        stopwords = json.loads(line)
        if 'ko' in stopwords:
            stopwords = stopwords['ko']
        else:
            stopwords = [] 
    return stopwords


def read_many_stop_words(fname):
    stopwords = []
    import many_stop_words as mstopwords
    stopwords = list(mstopwords.get_stop_words('kr'))
    return stopwords


STOPWORDS = {
    'morph': [
        {'name': 'bab2min', 'func': read_bab2min, 'file': 'raw/bab2min.txt'}
    ],
    'word': [
        {'name': '6', 'func': read_6 , 'file': 'raw/6.txt'},
        {'name': 'ranksnl', 'func': read_lined_stopwords, 'file': 'raw/ranksnl.txt'},
        {'name': 'spikeekips', 'func': read_lined_stopwords, 'file': 'raw/spikeekips.txt'},
        {'name': 'stopwords-iso', 'func': read_stopwords_iso, 'file': 'stopwords-iso/stopwords-iso.json'},
        {'name': 'many-stop-words', 'func': read_many_stop_words, 'file': None}
    ]
}


def read_stopwords(unit):
    global VERBOSE
    if VERBOSE > 0:
        print('{}\t{}\t{}\t{}'.format('UNIT', 'NAME', 'FUNC', 'FNAME'))

    all_stopwords = []
    for item in STOPWORDS[unit]:
        name, func, fname = item['name'], item['func'], item['file']
        stopwords = func(fname)
        all_stopwords.extend(stopwords)

        if VERBOSE > 0:
            print('{}\t{}\t{}\t{}'.format(unit, name, func, fname))
            print('{}'.format(', '.join(stopwords[:10])))

    all_stopwords = sorted(list(set(all_stopwords)))
    return all_stopwords


def dump_stopwords(stopwords, unit):
    fname = 'stopwords.{}.txt'.format(unit)
    with open(fname, 'w') as fp:
        fp.writelines('\n'.join(stopwords))


if __name__ == '__main__':
    '''
    여러 source에서 수집한 stopwords 파일들을 konlpy에서 사용할 형태로 가공합니다.
    
    stopwords의 형태는 unit에 따라 2가지로 나눠집니다.
    unit = morph or word

    두가지 unit을 한번에 처리하기 위해서는 unit을 all로 지정합니다.

    입력된 unit에 맞게 미리 지정된 경로에서 stopword를 읽어옵니다.
    중복된 stopword를 제거하고, 지정된 파일로 저장합니다.

    morph = stopwords.morph.txt
    word = stopwords.word.txt
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=0, help='set verbosity level')
    parser.add_argument('-u', '--unit', action='store', dest='unit', default='all', help='set update stopwords unit(morph | word | all(default))')
    options = parser.parse_args()

    VERBOSE = options.verbose
    unit = options.unit

    if unit not in ['all', 'morph', 'word']:
        print('UNIT is not acceptable', file=sys.stderr)
        sys.exit(1)

    if unit in ['all', 'morph']:
        stopwords = read_stopwords('morph')
        dump_stopwords(stopwords, 'morph')

    if unit in ['all', 'word']:
        stopwords = read_stopwords('word')
        dump_stopwords(stopwords, 'word')
