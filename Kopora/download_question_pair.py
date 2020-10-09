#!/usr/bin/env python3

from Korpora import Korpora, QuestionPairKorpus

question_pair = Korpora.load('question_pair')

print(question_pair.train[0])
print(question_pair.train[0].text)
print(question_pair.train[0].pair)
print(question_pair.train[0].label)

# print(question_pair.get_all_texts())
# print(question_pair.test.get_all_pairs())
print(type(question_pair))
print(question_pair.train[10])
for qpair in question_pair.train:
    print(qpair)
