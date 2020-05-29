stopwords
=====

konlpy에서 사용할 stopwords를 수집하고, 변환합니다.

### 제공하는 파일

- stopwords.morph.txt
  - 형태소 단위의 stopwords를 제공합니다.

- stopwords.word.txt
  - 단어 단위의 stopwords를 제공합니다.

### 파일 포맷

- stopwords.morph.txt

  ```
  morph1
  morph2
  ...
  ```

- stopwords.word.txt

  ```
  word1
  word2
  ...
  ```
 

### 출처

| file  | source  | unit |
|:------------- |:---------------|:-------------:|
| bab2min.txt      | [bab2min](https://bab2min.tistory.com/544) |  morph |
| ranksnl.txt      | [ranks.nl](https://www.ranks.nl/stopwords/korean) |  word |
| spikeekips.txt      | [spikeekips gist](https://gist.github.com/spikeekips/40eea22ef4a89f629abd87eed535ac6a) |  word |
| 6.txt      | [6 github](https://github.com/6/stopwords-json) |  word |
| stopwords-iso.txt      | [stopwords-iso github](https://github.com/stopwords-iso/stopwords-iso) |  word |
| many-stop-words.txt      | [many-stop-words](https://pypi.org/project/many-stop-words/) |  word |

- 각 stopwords는 수집된 source가 같은 경우가 있어 중복된 데이터가 있을 수 있습니다.
- 각 파일의 liecense는 아래와 같습니다.
- 일부 파일의 경우 저자에게 사용 허락을 받았습니다.
