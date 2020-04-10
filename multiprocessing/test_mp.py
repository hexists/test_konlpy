#!/usr/bin/env python3

'''
ref: 
https://github.com/konlpy/konlpy/issues/298
https://gist.github.com/wldh-g/117255b0d9cccf313120401db7a483d6

100, 1000개까지는 잘 실행되나, 10000개 실행시 오류 발생

multiprocessing.pool.RemoteTraceback:
"""
Traceback (most recent call last):
  File "/usr/local/Cellar/python/3.7.6_1/Frameworks/Python.framework/Versions/3.7/lib/python3.7/multiprocessing/pool.py", line 121, in worker
    result = (True, func(*args, **kwds))
  File "/usr/local/Cellar/python/3.7.6_1/Frameworks/Python.framework/Versions/3.7/lib/python3.7/multiprocessing/pool.py", line 44, in mapstar
    return list(map(*args))
  File "./test_mp.py", line 27, in morphRow
    result = kkma.morphs(doc)
  File "/usr/local/lib/python3.7/site-packages/konlpy/tag/_kkma.py", line 84, in morphs
    return [s for s, t in self.pos(phrase)]
  File "/usr/local/lib/python3.7/site-packages/konlpy/tag/_kkma.py", line 55, in pos
    sentences = self.jki.morphAnalyzer(phrase)
TypeError: No matching overloads found for kr.lucypark.kkma.KkmaInterface.morphAnalyzer(float), options are:
	public java.util.List kr.lucypark.kkma.KkmaInterface.morphAnalyzer(java.lang.String) throws java.lang.Exception
'''

import pandas as pd
import konlpy as kp
import multiprocessing as mp
import pdb
import sys
from pprint import pprint
from time import time

# Load Dataset
dataset = pd.read_csv('./nsmc/ratings_test.txt', sep='\t')
dataset = dataset.drop(['id'], axis=1)

# sample = dataset.iloc[10].document
# anal = kp.tag.Kkma().morphs(sample)
# pprint(anal)

kkma = None

def morphInit():
    global kkma
    kkma = kp.tag.Kkma()

def morphRow(doc):
    kkma = kp.tag.Kkma()
    result = kkma.morphs(doc)
    print('{}\t{}'.format(doc, result))
    return result

def morphRowDummy(doc):
    result = doc
    return result


print('start', file=sys.stderr)
start_time = time()

pool = mp.Pool(2, initializer=morphInit)
results = pool.map(morphRow, dataset.iloc[:10000, 0].values)
pool.close()

print('end', file=sys.stderr)
print(time() - start_time)
