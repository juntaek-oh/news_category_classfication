import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt, Komoran  # konlpy 자연어 처리 해주는 거 ( 형태소 분류기)
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
df = pd.read_csv('./crawling_data/naver_headline_total_250416.csv')
df.info()
print(df.head(30))
print(df.category.value_counts()) # 데이터의 양을 확인해서 늘릴지 줄일지 정함

X = df.titles
Y = df.category

print(X[1])
Okt = Okt()
# Okt_x = Okt.morphs(X[1], stem=True) # stem = 원형으로 바꿔주기
# print(Okt_x)
# Okt_x = Okt.morphs(X[1])
# print(Okt_x)
#
# komoran = Komoran()
# komoran_x = komoran.morphs(X[1])
# print(komoran_x)

encoder = LabelEncoder()
labeled_y = encoder.fit_transform(Y) # 라벨로 바꿔줌
print(labeled_y[:5])
label = encoder.classes_ # abcd 순으로 나눔
print(label) # encoder = ['Culture' 'Economic' 'IT' 'Politics' 'Social' 'World']
with open ('./models/encoder.pickle', 'wb') as f:  # 'wb' 바이너리 쓰기 모드
    pickle.dump(encoder, f) # encoder라는 객체를 encoder.pickle이라는 파일에 저장(dump = 직렬화)하는 작업입니다.
                             # pickle 은 어느 정보 형태에 상관없이 현재 형태로 저장된다.

onehot_y = to_categorical(labeled_y) # [1. 0. 0. 0. 0. 0.] 이런 모델로 바꿔줌 to_categorical
print(onehot_y)

# cleaned_x = re.sub('[^가-힣]', ' ', X[1]) #  ^ = 제외한  (가-힣)한글을 제외한것 ' '으로 대체
# print(X[1])
# print(cleaned_x)

for i in range(len(X)):  # 100개만 나눠줌                  100 자리에 len(X)
    X[i] = re.sub('[^가-힣]', ' ', X[i])
    X[i] = Okt.morphs(X[i], stem=True)
    if i % 1000 == 0: #1000개에 한번씩 프린트
        print(i)
print(X) # 형태소로 볼때 방해되는 한글자 등 없애기 전
for idx, sentence in enumerate(X):
    words = []
    for word in sentence:
        if len(word) > 1:  # 단어가 한글자 이상이면 추가
            words.append(word)
    X[idx] = ' '.join(words) # words에 빈칸 넣기

print(X) # 형태소로 볼때 방해되는거 다 제거한 후 결과

token = Tokenizer()   # 앞에 0 넣는 이유는 RNN이 기억을 잃어버리는 특성이 있어서 마지막에 0 넣으면 결과값이 0으로 나옴
token.fit_on_texts(X) # 형태소 받아서 토큰화 시키는 코드
tokened_x = token.texts_to_sequences(X) #형태소를 번호를 바꿔주는 코드
print(tokened_x) # 토큰화 시킨거
wordsize = len(token.word_index) + 1  # 라벨이 1에서 부터 시작 word_index =  총 형태소 갯수
print(wordsize) # 형태소 갯수 : 15396
max = 0
for sentence in tokened_x:
    if max < len(sentence):
        max = len(sentence)
print(max) # 25
with open('./models/token_max_{}.pickle'.format(max), 'wb') as f:
    pickle.dump(token, f)  # 피클담자



x_pad = pad_sequences(tokened_x, max) # max에 맞는 길이 맞춰주기  [ 0  0 20 21 22 23  4 24 25 26] 앞에 0 채우기
print(x_pad) #  [   0    0    0 ... 2359  279 1936]]

x_train, x_test, y_train, y_test = train_test_split(
    x_pad, onehot_y, test_size=0.1)  # 표본의 0.9 의 트레인 0.1 의 테스트 나눔
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

np.save('./crawling_data/title_x_train_wordsize{}'.format(wordsize), x_train)
np.save('./crawling_data/title_x_test_wordsize{}'.format(wordsize), x_test)
np.save('./crawling_data/title_y_train_wordsize{}'.format(wordsize), y_train)
np.save('./crawling_data/title_y_test_wordsize{}'.format(wordsize), y_test)