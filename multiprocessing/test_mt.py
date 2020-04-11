#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from konlpy.tag import Kkma
from konlpy.corpus import kolaw
from threading import Thread
import jpype
import pandas as pd


def do_concurrent_tagging(start, end, lines, result):
    jpype.attachThreadToJVM()
    l = [k.pos(lines[i]) for i in range(start, end)]
    result.append(l)
    return

if __name__=="__main__":
    import time

    print('Number of lines in document:')
    k = Kkma()

    # Load Dataset
    # dataset = pd.read_csv('../nsmc/ratings_test.txt', sep='\t')
    dataset = pd.read_csv('ratings_test.txt', sep='\t')
    # dataset = pd.read_csv('sample', sep='\t')
    dataset = dataset.drop(['id'], axis=1)
    
    # fiter NaN
    # https://stackoverflow.com/a/61125923
    filt_condition = dataset['document'].str.contains("", na=False)
    dataset = dataset[filt_condition]

    lines = dataset.iloc[:, 0].values
    nlines = len(lines)
    print(nlines)

    print('Batch tagging:')
    s = time.time()
    result = []
    # l = [k.pos(line) for line in lines]
    l = []
    for line in lines:
        print(line)
        anal = k.pos(line)
        print(anal)
        l.append(anal)
    result.append(l)
    t = time.time()
    print(t - s)

    print('Concurrent tagging:')
    result = []
    t1 = Thread(target=do_concurrent_tagging, args=(0, int(nlines/2), lines, result))
    t2 = Thread(target=do_concurrent_tagging, args=(int(nlines/2), nlines, lines, result))
    t1.start(); t2.start()
    t1.join(); t2.join()

    m = sum(result, []) # Merge results
    print(time.time() - t)
