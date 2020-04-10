#! /usr/bin/python3
# -*- coding: utf-8 -*-

'''
ref: http://konlpy.org/ko/latest/examples/multithreading/

공식 문서에 있는 설명, python3에서 time.clock()를 지원하지 않아 time.time()으로 변경
'''


from konlpy.tag import Kkma
from konlpy.corpus import kolaw
from threading import Thread
import jpype

def do_concurrent_tagging(start, end, lines, result):
    jpype.attachThreadToJVM()
    l = [k.pos(lines[i]) for i in range(start, end)]
    result.append(l)
    return

if __name__=="__main__":
    import time

    print('Number of lines in document:')
    k = Kkma()

    with open('sample') as fp:
        lines = fp.read().splitlines()
    lines = lines[:300]
    nlines = len(lines)
    print(nlines)

    print('Batch tagging:')
    s = time.time()
    result = []
    l = [k.pos(line) for line in lines]
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
