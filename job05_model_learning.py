import numpy as np
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *


x_train = np.load('./crawling_data/title_x_train_wordsize15396.npy', allow_pickle=True)
x_test = np.load('./crawling_data/title_x_test_wordsize15396.npy', allow_pickle=True)
y_train = np.load('./crawling_data/title_y_train_wordsize15396.npy', allow_pickle=True)
y_test = np.load('./crawling_data/title_y_test_wordsize15396.npy', allow_pickle=True)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

model = Sequential()
model.add(Embedding(15396, 300)) # 형태소 각각을 고정된 길이의 숫자 벡터로 바꿔주는 함수입니다.
# 차원이 늘어나면 늘어날수록 희소해져서 학습이 안됨 그래서 300으로 차원 축소를 시킴 축소 시키면 데이터 손실이 일어남
model.build(input_shape = (None, 25)) # max 가 25
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu')) # kernel = 잘라서 보겠다.
model.add(MaxPool1D(pool_size=1)) #  pool_size = x * x 의 크기 중에 가장 큰값 오래걸리니까  되는지 실험 할때 씀
model.add(LSTM(128, activation='tanh', return_sequences=True)) #Long Short-Term Memory(LSTM) activation tanh고정 , -1~1
model.add(Dropout(0.3))                                        # 섭씨 화씨는 마지막에 activation 안씀
model.add(LSTM(64, activation='tanh', return_sequences=True)) # LSTM을 실행하면 결과 값이 1개라 return_sequences 사용해서
model.add(Dropout(0.3))                                       # 결과값이 입력과 같아서 LSTM에서 다시 되먹임 되게 해줌
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer= 'adam',
              metrics=['accuracy'])

fit_hist = model.fit(x_train, y_train, batch_size =128,
                     epochs =10, validation_data =(x_test, y_test))
score = model.evaluate(x_test,y_test, verbose=0)
print('Final test set accuracy', score[1])
model.save('./models/news_section_classfication_model_{}.h5'.format(score[1]))
plt.plot(fit_hist.history['val_accuracy'], label='val_accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.legend()
plt.show()