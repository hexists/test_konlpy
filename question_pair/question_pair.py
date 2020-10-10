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
  e) custom dataset 생성
  f) model 생성
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

print('vocab2idx = {}'.format(len(vocab2idx)))
print('idx2vocab = {}'.format(idx2vocab[:10]))
print()


import torch
import torch.utils.data as data_utils
import torch.nn.utils.rnn as rnn
import numpy as np
import random


random_seed = 111
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


class QuestionPairDataset(torch.utils.data.Dataset): 
    def __init__(self, vocab2idx, data):
        self.vocab2idx = vocab2idx
        self.data = self.text2idx(data)

    def text2idx(self, pairs):
        # train, test 데이터를 index형태로 변환합니다.
        # text_morph, pair_morph에 대해 index로 변환하고, 다른 데이터는 그대로 유지합니다.
        question_pairs = []
        for text, pair, label, text_morph, pair_morph in pairs:
            text_idx, pair_idx = [], []
            for morph in text_morph:
                idx = self.vocab2idx[morph] if morph in self.vocab2idx else self.vocab2idx['<unk>']
                text_idx.append(idx)
    
            for morph in pair_morph:
                idx = self.vocab2idx[morph] if morph in self.vocab2idx else self.vocab2idx['<unk>']
                pair_idx.append(idx)
    
            text_idx = torch.LongTensor(text_idx)
            pair_idx = torch.LongTensor(pair_idx)
            label = torch.FloatTensor([float(label)])

            question_pairs.append((text, pair, label, text_idx, pair_idx))
            # pprint(question_pairs)

        return question_pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, pair, label, text_idx, pair_idx = self.data[idx]
        text_len, pair_len = len(text_idx), len(pair_idx)
        return text, pair, text_idx, pair_idx, label, text_len, pair_len

train_dataset = QuestionPairDataset(vocab2idx, train)
test_dataset = QuestionPairDataset(vocab2idx, test)

valid_len = int(len(train_dataset) * 0.2)
train_dataset, valid_dataset = data_utils.random_split(train_dataset, (len(train_dataset) - valid_len, valid_len))

def make_batch(samples):
    text, pair, text_idx, pair_idx, label, text_len, pair_len = list(zip(*samples))

    text_len = torch.LongTensor(text_len)
    pair_len = torch.LongTensor(pair_len)

    padded_text_idx = rnn.pad_sequence(text_idx, batch_first=True, padding_value=0)
    padded_pair_idx = rnn.pad_sequence(pair_idx, batch_first=True, padding_value=0)

    batch = [
        text, 
        pair,
        padded_text_idx.contiguous(),
        padded_pair_idx.contiguous(),
        torch.stack(label).contiguous(),
        text_len,
        pair_len
    ]
    return batch


batch_size = 2
train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=make_batch)
valid_loader = data_utils.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=make_batch)

# for i, vals in enumerate(train_loader):
#     print(vals)


