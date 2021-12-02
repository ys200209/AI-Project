from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import tensorflow as tf


seed = 0 # seed 값 설정 (동일한 랜덤값을 추출해주기 위해서 시드값을 설정해준다)
np.random.seed(seed)
tf.random.set_seed(3)

df = pd.read_csv("./dataset/housing.csv", delim_whitespace=True,
    header=None) # 공백으로 구분된 값으로 데이터 파일을 읽어온다.

dataset = df.values
X = dataset[:, 0:13] # 13개의 피쳐를 X 변수에 담는다.
Y = dataset[:, 13] # 한개의 class를 Y 변수에 담는다.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)
# 전체 데이터 셋중에서 70%만 train 데이터로 사용하고 30%는 테스트를 하는데 쓰이도록 설정

model = Sequential()
model.add(Dense(30, input_dim=13, activation="relu")) # 13개의 input과 30개의 output으로 구성된 입력층 생성
model.add(Dense(6, activation="relu")) # 6개의 input과 1개의 output으로 구성된 은닉층 생성
model.add(Dense(1)) # 선형회귀는 출력층에 활성화함수를 입력하지 않는다.

model.compile(loss="mean_squared_error", optimizer="adam") # 오차 함수를 평균오차제곱으로, 최적화 함수를 adam으로 설정

model.fit(X_train, Y_train, epochs=200, batch_size=10) 
# 전체 데이터의 70%를 훈련 데이터로 사용한다.  # 각 샘플을 200번 반복 훈련하며 한번 훈련당 10번씩 훈련한다.

# 예측 값과 실제 값의 비교
Y_prediction = model.predict(X_test).flatten()
for i in range(10):
    label = Y_test[i] # 실제값
    prediction = Y_prediction[i] # 예측값
    print("실제가격: {:.3f}, 예상가격: {:.3f}".format(label, prediction))
