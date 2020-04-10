#!/usr/bin/env python3


from konlpy.tag import Hannanum
hannanum = Hannanum()

print('[Hannanum]')
print(hannanum.analyze('롯데마트의 흑마늘 양념 치킨이 논란이 되고 있다.'))

from konlpy.tag import Kkma
kkma = Kkma()
print('[Kkma]')
print(kkma.morphs('공부를 하면할수록 모르는게 많다는 것을 알게 됩니다.'))
