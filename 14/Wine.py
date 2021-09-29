from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

import pandas as pd
import numpy
import tensorflow as tf
import matplotlib.pyplot as plt

# seed 값 설정
# seed = 0
numpy.random.seed(3)
tf.random.set_seed(3)

# 데이터 입력
df_pre = pd.read_csv('./dataset/wine.csv', header=None)
df = df_pre.sample(frac=0.15)
dataset = df.values
X = dataset[:, 0:12]
Y = dataset[:, 12]

# 모델 설정
model = Sequential()
model.add(Dense(30, input_dim=12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# 모델 컴파일
model.compile(loss="binary_crossentropy", 
        optimizer="adam",
        metrics=["accuracy"])

# 학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor="val_loss", patience=100)

# 모델 실행
# model.fit(X, Y, epochs=200, batch_size=200)
history = model.fit(X, Y, validation_split=0.2, epochs=3500, 
    batch_size=500, callbacks=[early_stopping_callback])

# y_vloss에 테스트셋으로 실험 결과의 오차 값을 저장
y_vloss = history.history['val_loss']

# y_acc에 학습셋으로 측정한 정확도의 값을 저장
y_acc = history.history['accuracy']

# x 값을 지정하고 정확도를 파란색으로, 오차를 빨간색으로 표시
x_len = numpy.arange(len(y_acc))
plt.plot(x_len, y_vloss, "o", c="red", markersize=3)
plt.plot(x_len, y_acc, "o", c="blue", markersize=3)

plt.show()


# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))