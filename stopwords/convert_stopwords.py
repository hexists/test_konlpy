#!/usr/bin/env python3


import os
import sys
import json
import argparse


def read_bab2min(fname):
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


STOPWORDS_PATHS = {
    'morph': {'bab2min': read_bab2min},
    'word': {'bab2min': read_bab2min}
}


def read_stopwords(unit):
    stopwords = []
    for path in STOPWORDS_PATHS[unit]:
        stopwords = read_bab2min()


def dumps_stopwords(stopwords, unit):
    fname = 'stopwords.{}.txt'.format(unit)
    with open(fname, 'w') as fp:
        fp.writelines('\n'.join(stopwords))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--unit', action='store', dest='unit', required=True, help='set update stopwords unit')
    options = parser.parse_args()

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
