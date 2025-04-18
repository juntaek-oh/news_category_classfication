import pickle
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from konlpy.tag import Okt
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import re


df = pd.read_csv('./crawling_data/naver_headline_news_250418.csv')
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.head())
df.info()
print(df.category.value_counts())

X = df.titles
Y = df.category

with open('./models/encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)    # 넣을땐 dump , 부를떈 load
label = encoder.classes_
print(label)    # 라벨링하고 전처리 했을때랑 똑같이 해줘야 함  안그러면 결과값이 아예 달라짐

labeled_y = encoder.transform(Y)  # fit_transform = 라벨링을 해주고 바꿔주는거 , 이미 정보가 있는라벨은 transform
onehot_y = to_categorical(labeled_y)
print(onehot_y)

okt = Okt()
for i in range(len(X)):
    X[i] = re.sub('[^가-힣]', ' ', X[i])
    X[i] = okt.morphs(X[i],stem=True)
print(X[:10])

for idx, sentence in enumerate(X):
    words = []
    for word in sentence:
        if len(word) > 1:  # 단어가 한글자 이상이면 추가
            words.append(word)
    X[idx] = ' '.join(words) #

print(X[:10])


#앞에 0 넣는 이유는 RNN이 기억을 잃어버리는 특성이 있어서 마지막에 0 넣으면 결과값이 0으로 나옴
with open('./models/token_max_25.pickle', 'rb') as f:
    token = pickle.load(f)
tokened_x = token.texts_to_sequences(X)
print(tokened_x[:5])

for i in range(len(tokened_x)):
    if len(tokened_x[i]) > 25:
        tokened_x[i] = tokened_x[i][:25]
x_pad = pad_sequences(tokened_x, 25)
print(x_pad)

model = load_model('./models/news_section_classfication_model_0.7355035543441772.h5')
preds = model.predict(x_pad)
print(preds)

predict_section = []
for pred in preds:
    # predict_section.append(label[np.argmax(pred)]) 하나 확인 하는 코드
    most = label[np.argmax(pred)]
    pred[np.argmax(pred)] = 0
    second =  label[np.argmax(pred)]
    predict_section.append([most, second])
print(predict_section)

df['predict'] = predict_section
print(df[['category', 'predict']].head(30))

score = model.evaluate(x_pad, onehot_y)
print(score[1])     # 0.587

df['OX'] = 0
for i in range(len(df)):
    if df.loc[i, 'category'] in df.loc[i, 'predict']:
        df.loc[i, 'OX'] = 1
print(df.OX.mean())