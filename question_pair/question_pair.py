#!/usr/bin/env python3

'''
korpora + konlpy + pytorch를 이용한 question pair 문제 풀기

https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07
긅을 참고했습니다.

1) korpora, konlpy, pytorch 설치하기

  $ pip install -r requirments.txt

  $ bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)

2) 잘 설치됐는지 테스트 해보기
  # Kopora
  $ python3
  >>> from Korpora import Korpora, QuestionPairKorpus
  >>> question_pair = Korpora.load('question_pair')
  
      Korpora 는 다른 분들이 연구 목적으로 공유해주신 말뭉치들을
      손쉽게 다운로드, 사용할 수 있는 기능만을 제공합니다.
  
      말뭉치들을 공유해 주신 분들에게 감사드리며, 각 말뭉치 별 설명과 라이센스를 공유 드립니다.
      해당 말뭉치에 대해 자세히 알고 싶으신 분은 아래의 description 을 참고,
      해당 말뭉치를 연구/상용의 목적으로 이용하실 때에는 아래의 라이센스를 참고해 주시기 바랍니다.
  
      # Description
       Author : songys@github
      Repository : https://github.com/songys/Question_pair
      References :
  
      질문쌍(Paired Question v.2)
      짝 지어진 두 개의 질문이 같은 질문인지 다른 질문인지 핸드 레이블을 달아둔 데이터
      사랑, 이별, 또는 일상과 같은 주제로 도메인 특정적이지 않음
  
      # License
      Creative Commons Attribution-ShareAlike license (CC BY-SA 4.0)
      Details in https://creativecommons.org/licenses/by-sa/4.0/
  
  >>> print(question_pair.train[0])
  LabeledSentencePair(text='1000일 만난 여자친구와 이별', pair='10년 연예의끝', label='1')
  
  >>> print(question_pair.train[0].text)
  1000일 만난 여자친구와 이별
  >>> print(question_pair.train[0].pair)
  10년 연예의끝
  >>> print(question_pair.train[0].label)
  1

  # konlpy mecab
  $ python3
  >>> from konlpy.tag import Mecab
  >>> mecab = Mecab()
  >>> print(mecab.pos('가을 하늘 공활한데 높고 구름 없이'))
  [('가을', 'NNG'), ('하늘', 'NNG'), ('공활', 'XR'), ('한', 'XSA+ETM'), ('데', 'NNB'), ('높', 'VA'), ('고', 'EC'), ('
  구름', 'NNG'), ('없이', 'MAG')]

3) step by step

  a) 데이터 로드
  b) 형태소 분석
  c) vocab 생성
  d) idx로 변환
  e) custom dataset 생성
  f) model 생성
  g) model 학습
  h) 학습 완료 후 test 수행
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
            # label = torch.FloatTensor(float(label))
            label = torch.as_tensor(float(label))

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


batch_size = 64
train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=make_batch)
valid_loader = data_utils.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=make_batch)
test_loader = data_utils.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=make_batch)

# for i, vals in enumerate(train_loader):
#     print(vals)


import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch import optim
from torch.utils.tensorboard import SummaryWriter


class TextSiamese(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(TextSiamese, self).__init__()
        self.embed_size = 300
        self.num_layers = 1
        self.bidirectional = True
        self.direction = 2 if self.bidirectional else 1
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embeddings = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=0)

        self.shared_lstm = nn.LSTM(self.embed_size, self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirectional)

    def forward(self, text, pair, text_len, pair_len):
        pack_text = self.input2packed_embed(text, text_len)
        pack_pair = self.input2packed_embed(pair, pair_len)

        batch_size = text.size()[0]

        # use zero init hidden, cell_state
        _, (text_hidden, text_cell_state) = self.shared_lstm(pack_text, None)
        _, (pair_hidden, pair_cell_state) = self.shared_lstm(pack_pair, None)

        if self.bidirectional is True:
            text_hidden = text_hidden.view(self.num_layers, self.direction, batch_size, self.hidden_size)  # (num_layers, num_directions, batch, hidden_size)
            text_hidden = torch.cat((text_hidden[:, 0], text_hidden[:, 1]), -1)  # (num_directions, batch, hidden_size) => (num_directions, batch, hidden_size * 2)
            pair_hidden = pair_hidden.view(self.num_layers, self.direction, batch_size, self.hidden_size)
            pair_hidden = torch.cat((pair_hidden[:, 0], pair_hidden[:, 1]), -1)

        distance = self.exponent_neg_manhattan_distance(text_hidden.permute(1, 2, 0).view(batch_size, -1), pair_hidden.permute(1, 2, 0).view(batch_size, -1))  # (batch_size, hidden_size)

        return distance

    def input2packed_embed(self, inp, inp_len):
        embed = self.embeddings(inp)  # (Batch size, Max length, Embedding size)
        packed_embed = pack_padded_sequence(embed, inp_len, batch_first=True, enforce_sorted=False)
        return packed_embed

    def exponent_neg_manhattan_distance(self, x1, x2):
        ''' Helper function for the similarity estimate of the LSTMs outputs '''
        return torch.exp(-torch.sum(torch.abs(x1 - x2), dim=1))  # (batch_size)

hidden_size = 50
learning_rate = 0.001
num_iters = 1000

model = TextSiamese(hidden_size, len(vocab2idx))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.MSELoss()

def print_progress(msg, progress):
    max_progress = int((progress*100)/2)
    remain=50-max_progress
    buff="{}\t[".format( msg )
    for i in range( max_progress ): buff+="⬛"
    buff+="⬜"*remain
    buff+="]:{:.2f}%\r".format( progress*100 )
    sys.stderr.write(buff)

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

from datetime import datetime
date_time = datetime.now().strftime('%Y%m%d%H%M')
writer = SummaryWriter('./runs/{}'.format(date_time))

for epoch in range(1, num_iters + 1):
    model.train()
    tr_losses, tr_accs = [], []
    # train
    for i, vals in enumerate(train_loader):
        text, pair, text_idx, pair_idx, label, text_len, pair_len = vals

        model.zero_grad()
        scores = model(text_idx, pair_idx, text_len, pair_len)
        loss = loss_func(scores, label)
        acc = binary_acc(scores, label)

        tr_losses.append(loss.item())
        tr_accs.append(acc.item())

        loss.backward()
        optimizer.step()

        progress = (i + 1) / float(len(train_loader))
        print_progress('train', progress)
    print(file=sys.stderr)

    with torch.no_grad():
        model.eval()
        va_losses, va_accs = [], []
        for i, vals in enumerate(valid_loader):
            text, pair, text_idx, pair_idx, label, text_len, pair_len = vals

            scores = model(text_idx, pair_idx, text_len, pair_len)
            loss = loss_func(scores, label)
            acc = binary_acc(scores, label)

            va_losses.append(loss.item())
            va_accs.append(acc.item())

            progress = (i + 1) / float(len(valid_loader))
            print_progress('valid', progress)

        print(file=sys.stderr)
        print('text : {}'.format(text[-1]), file=sys.stderr)
        print('pair : {}'.format(pair[-1]), file=sys.stderr)
        print('label = {:.4f}, score = {:.4f}'.format(label[-1].item(), scores[-1].item()), file=sys.stderr)

    tr_loss, tr_acc = np.mean(tr_losses), np.mean(tr_accs)
    va_loss, va_acc = np.mean(va_losses), np.mean(va_accs)

    print("{} / {}\ttrain loss : {:.4f}, train acc: {:.4f}, valid loss: {:.4f} valid acc: {:.4f}\n".format(epoch, num_iters, tr_loss, tr_acc, va_loss, va_acc), file=sys.stderr)

    writer.add_scalar('{}/{}'.format('loss', 'train'), tr_loss, epoch)
    writer.add_scalar('{}/{}'.format('acc', 'train'), tr_acc, epoch)
    writer.add_scalar('{}/{}'.format('loss', 'valid'), va_loss, epoch)
    writer.add_scalar('{}/{}'.format('acc', 'valid'), va_acc, epoch)

writer.close()

print(file=sys.stderr)
with torch.no_grad():
    model.eval()
    cor_count, fp = 0, open('log.{}'.format(date_time), 'w')
    fp.writelines('{}\t{}\t{}\t{}\t{}\n'.format('T/F', 'LABEL', 'PRED', 'TEXT', 'PAIR'))
    for i, vals in enumerate(test_loader):
        text, pair, text_idx, pair_idx, label, text_len, pair_len = vals

        scores = model(text_idx, pair_idx, text_len, pair_len)
        pred_label = torch.round(scores)
        if pred_label == label:
            cor_count += 1

        fp.writelines('{}\t{}\t{}\t{}\t{}\n'.format((pred_label == label)[0].item(), label[0].item(), pred_label[0].item(), text[0], pair[0]))

        progress = (i + 1) / float(len(test_loader))
        print_progress('test', progress)

    te_acc = cor_count / i
    print("\ntest acc: {:.4f}".format(te_acc), file=sys.stderr)
    fp.close()
