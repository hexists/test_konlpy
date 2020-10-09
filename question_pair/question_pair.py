#!/usr/bin/env python3

'''
kopora + konlpy + pytorch를 이용한 question pair 문제 풀기

https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07
긅을 참고했습니다.

1) kopora, konlpy, pytorch 설치하기

  $ pip install -r requirments.txt

  $ bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)

2) 잘 설치됐는지 테스트 해보기

  # Kopora
  $ from Korpora import Korpora, QuestionPairKorpus

  $ question_pair = Korpora.load('question_pair')

  $ print(question_pair.train[0])
  $ print(question_pair.train[0].text)
  $ print(question_pair.train[0].pair)
  $ print(question_pair.train[0].label)

  # konlpy mecab
  $ from konlpy.tag import Mecab
    
  $ mecab = Mecab()
    
  $ print(mecab.pos('가을 하늘 공활한데 높고 구름 없이'))

3) step by step

  a) 데이터 로드
  b) 형태소 분석
  c) vocab 생성
  d) idx로 변환
'''


import os
import sys
from pprint import pprint


# Kopora에서 question pair corpus를 불러옵니다.
from Korpora import Korpora
question_pair = Korpora.load('question_pair')

# 불러온 corpus를 형태소 분석합니다.
# 형태소 분석은 konlpy mecab을 이용합니다.
# konlpy의 mecab을 불러오고 초기화합니다.
from konlpy.tag import Mecab
mecab = Mecab()

# 형태소 분석
def get_morph_tag(text):
    morphs, tags = [], []
    for morph, tag in mecab.pos(text):
        morphs.append(morph)
        tags.append(tag)
    return morphs, tags

# train, test 데이터를 제공하고 있습니다.
# 데이터는 LabeldSetencePair Object로 제공되며, text, pair, label이 있습니다.
# LabeledSentencePair(text='1000일 만난 여자친구와 이별', pair='10년 연예의끝', label='1')

def analyze_question_pairs(question_pairs):
    anal_question_pairs = []
    for qp in question_pairs:
        # print('text  = {}'.format(qp.text))
        # print('pair  = {}'.format(qp.pair))
        # print('label = {}'.format(qp.label))

        text_morph, _ = get_morph_tag(qp.text)
        pair_morph, _ = get_morph_tag(qp.pair)

        anal_question_pairs.append((qp.text, qp.pair, qp.label, text_morph, pair_morph))
        # pprint(anal_question_pairs)
    return anal_question_pairs

train = analyze_question_pairs(question_pair.train)
test = analyze_question_pairs(question_pair.test)

# train 데이터를 이용하여 vocab을 만듭니다.
# <pad>, <unk>에 대해서는 미리 index를 지정합니다.
vocab2idx = {'<pad>': 0, '<unk>': 1}
idx2vocab = ['<pad>', '<unk>']

for _, _, _, text_morph, pair_morph in train:
    for morph in text_morph:
        if morph not in vocab2idx:
            vocab2idx[morph] = len(idx2vocab)
            idx2vocab.append(morph)

pprint('vocab2idx = {}'.format(len(vocab2idx)))
pprint(idx2vocab[:10])
print()

# train, test 데이터를 index형태로 변환합니다.
# text_morph, pair_morph에 대해 index로 변환하고, 다른 데이터는 그대로 유지합니다.
def text2idx(pairs):
    question_pairs = []
    for text, pair, label, text_morph, pair_morph in pairs:
        text_idx, pair_idx = [], []
        for morph in text_morph:
            idx = vocab2idx[morph] if morph in vocab2idx else vocab2idx['<unk>']
            text_idx.append(idx)

        for morph in pair_morph:
            idx = vocab2idx[morph] if morph in vocab2idx else vocab2idx['<unk>']
            pair_idx.append(idx)

        question_pairs.append((text, pair, label, text_idx, pair_idx))
        pprint(question_pairs)
        break
    return question_pairs

train = text2idx(train)
test = text2idx(test)


