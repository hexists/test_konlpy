#!/usr/bin/env python3


from konlpy.tag import Hannanum, Kkma, Komoran, Mecab, Okt

hannanum = Hannanum()
print('[Hannanum]')
print(hannanum.analyze('롯데마트의 흑마늘 양념 치킨이 논란이 되고 있다.'))

kkma = Kkma()
print('[Kkma]')
print(kkma.morphs('공부를 하면할수록 모르는게 많다는 것을 알게 됩니다.'))

komoran = Komoran()
print('[Komoran]')
print(komoran.morphs(u'우왕 코모란도 오픈소스가 되었어요'))

mecab = Mecab()
print('[Mecab]')
print(mecab.morphs(u'영등포구청역에 있는 맛집 좀 알려주세요.'))

okt = Okt()
print('[Okt]')
print(okt.morphs(u'단독입찰보다 복수입찰의 경우'))
