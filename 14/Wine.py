from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy
import tensorflow as tf
import matplotlib.pyplot as plt

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(3)

# 데이터 입력
df_pre = pd.read_csv("./dataset/wine.csv", header=None) # 데이터 셋을 불러오는 함수
df = df_pre.sample(frac=1) # wine.csv 파일의 전체 샘플(=1)을 사용하도록 frac 파라미터에 값을 줌
dataset = df.values
X = dataset[:, 0:12] # 12개의 피쳐를 X값에 담는 부분
Y = dataset[:, 12] # 마지막 클래스를 Y값에 담는 부분

# 학습 데이터와 테스트 데이터를 분리
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)
# train 데이터 셋은 전체 데이터의 70% 이며, test 데이터 셋은 나머지 30% 이다.

# 모델 설정
model = Sequential()
model.add(Dense(30, input_dim=12, activation="relu")) # 12개의 input과 30개의 output을 가지는 입력층을 생성
model.add(Dense(12, activation="relu")) # 12개의 input과 8개의 output을 가지는 은닉층을 생성
model.add(Dense(8, activation="relu")) # 8개의 input과 1개의 output을 가지는 두 번째 은닉층을 생성
model.add(Dense(1, activation="sigmoid")) # 1개의 input을 가지고 결과를 출력하는 출력층을 생성

# 모델 컴파일
model.compile(loss="binary_crossentropy", # 오차 함수로 binary_crossentropy 함수를 이용
        optimizer="adam", # 최적화 함수로 adam 함수를 이용
        metrics=["accuracy"]) # 결과 출력을 accuracy 형태로 나타냄

# 모델 실행
model.fit(X_train, Y_train, epochs=200, batch_size=200)
# 전체 데이터 셋의 70%를 학습 데이터로 이용했으며 각 샘플당 200번 학습하고 한번 학습에 모든 샘플을 동시에 학습시켰다.

# 결과 출력
print("\n Accruacy: %.4f" % (model.evaluate(X_test, Y_test)[1])) # 나머지 30% 테스트 데이터의 정확성에 대한 지표.
