#!/usr/bin/env python3

from Korpora import Korpora, QuestionPairKorpus
from konlpy.tag import Mecab

question_pair = Korpora.load('question_pair')
mecab = Mecab()


train_fn = 'qqp.train'
test_fn = 'qqp.test'


def get_morph_tag(text):
    morphs, tags = [], []
    for morph, tag in mecab.pos(text):
        morphs.append(morph)
        tags.append(tag)
    return morphs, tags


def analyze_and_save(fn, df):
    fp = open(fn, 'w')
    for idx, qpair in enumerate(df):
        text_morphs, text_tags = get_morph_tag(qpair.text)
        pair_morphs, pair_tags = get_morph_tag(qpair.pair)
        text_morphs, text_tags = ' '.join(text_morphs), ' '.join(text_tags)
        pair_morphs, pair_tags = ' '.join(pair_morphs), ' '.join(pair_tags)
        fp.writelines('{}\n'.format('\t'.join([qpair.text, qpair.pair, qpair.label, text_morphs, text_tags, pair_morphs, pair_tags])))
    fp.close()

analyze_and_save(train_fn, question_pair.train)
analyze_and_save(test_fn, question_pair.test)
