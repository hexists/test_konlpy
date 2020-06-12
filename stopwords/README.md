stopwords
=====

konlpy에서 사용할 stopwords를 수집하고, 변환합니다.

### 제공하는 파일

- stopwords.morph.txt
  - 형태소 단위의 stopwords를 제공합니다.

- stopwords.word.txt
  - 단어 단위의 stopwords를 제공합니다.

### 파일 포맷

- stopwords.morph.txt: 분석기별로 stopwords가 저장되어 있습니다.

  ```
  analyzer1 \t analyzer \t ...
  morph1    \t morph1   \t ...
  morph2    \t morph2   \t ...
  ...
  ```

- stopwords.word.txt

  ```
  word1
  word2
  ...
  ```

### raw 파일 준비하기

파일 저장, git clone, pypi로 설치합니다.

- 파일로 저장하기
  - [bab2min](https://bab2min.tistory.com/544)에 있는 내용을 test_konlpy/stopwords/raw/bab2min.txt에 저장합니다.
  - [ranks.nl](https://www.ranks.nl/stopwords/korean)에 있는 내용을 test_konlpy/stopwords/raw/ranksnl.txt에 저장합니다.
  - [spikeekips gist](https://gist.github.com/spikeekips/40eea22ef4a89f629abd87eed535ac6a)에 있는 내용을 test_konlpy/stopwords/raw/spikeekips.txt에 저장합니다.
  - [6 github](https://github.com/6/stopwords-json/blob/master/dist/ko.json)에 있는 내용을 test_konlpy/stopwords/raw/6.txt에 저장합니다.

- git clone 받기

  ```
  $ cd /path/to/test_konlpy/stopwords
  
  $ git clone https://github.com/stopwords-iso/stopwords-iso
  ```

- pypi로 설치하기

  ```
  $ pip install many-stop-words
  ```

### 변환하기

stopwords 디렉토리로 이동 후 변환 프로그램을 수행합니다.
생성하고자 하는 stopwords에 따라 unit(-u | --unit)을 지정합니다.

- morph: 형태소 단위 stopwords 생성
- word: 단어 단위 stopwords 생성

```
$ cd /path/to/test_konlpy/stopwords

$ python3 convert_stopwords.py -u morph
   or
   python3 convert_stopwords.py -u word
   or
   python3 convert_stopwords.py -u all
```

```
$ python3 convert_stopwords.py --help

usage: convert_stopwords.py [-h] [-v] [-u UNIT] [-t C_TABLE_FILE]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         set verbosity level
  -u UNIT, --unit UNIT  set update stopwords unit(morph | word | all(default))
  -t C_TABLE_FILE, --conv-table C_TABLE_FILE
                        set convert table path for stopwords.morphs.txt
```

### 출처

| file  | source  | unit | license |
|:------------- |:---------------|:-------------:|:--:|
| bab2min.txt      | [bab2min](https://bab2min.tistory.com/544) |  morph | 저자에게 사용 허락 받음 | 
| ranksnl.txt      | [ranks.nl](https://www.ranks.nl/stopwords/korean) |  word | MIT로 추정 | 
| spikeekips.txt      | [spikeekips gist](https://gist.github.com/spikeekips/40eea22ef4a89f629abd87eed535ac6a) |  word |저자에게 사용 허락 받음 | 
| 6.txt      | [6 github](https://github.com/6/stopwords-json) |  word |apache 2.0 | 
| stopwords-iso.txt      | [stopwords-iso github](https://github.com/stopwords-iso/stopwords-iso) |  word |MIT | 
| many-stop-words.txt      | [many-stop-words](https://pypi.org/project/many-stop-words/) |  word |MIT | 

- 각 stopwords는 수집된 source가 같은 경우가 있어 중복된 데이터가 있을 수 있습니다.
- 일부 파일의 경우 저자에게 사용 허락을 받았습니다.
- ranksnl 데이터는 stopwords-iso, many-stop-words에서 공개되어 있어 MIT License로 추정됩니다.
