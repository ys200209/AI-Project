from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import numpy
import pandas as pd
import tensorflow as tf

seed = 0 # seed 값 설정 (동일한 랜덤값을 추출하기 위해 랜덤 시드값을 설정해준다.)
numpy.random.seed(seed)
tf.random.set_seed(seed)

df = pd.read_csv("./dataset/sonar.csv", header=None) # csv파일을 불러오는 함수

dataset = df.values
X = dataset[:, 0:60].astype(float) # 입력 피쳐들을 float 타입으로 변환
Y_obj = dataset[:, 60]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj) # 문자열로 된 class 데이터를 머신러닝이 학습할 수 있는 숫자형태로 변환해주는 코드

# 10개의 파일로 쪼개어 각각을 교차검증하는 코드
n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

# 빈 accuracy 배열
accuracy = []

# 모델의 설정, 컴파일, 실행
for train, test in skf.split(X, Y):

    model = Sequential()
    model.add(Dense(24, input_dim=60, activation="relu"))
    # 60개의 input과 24개의 output으로 구성된 입력층 생성
    model.add(Dense(10, activation="relu")) # 10개의 input과 1개의 output으로 구성된 은닉층 생성
    model.add(Dense(1, activation="sigmoid")) # 1개의 input으로 구성된 출력층 생성 (활성화 함수 : 시그모이드)

    model.compile(loss="mean_squared_error", # 오차 함수로 평균 제곱 오차 함수를 사용하고
            optimizer="adam", # 최적화 함수로 adam 함수를 사용하고
            metrics=["accuracy"]) # 결과값을 accuracy 형태로 보여지도록 모델을 컴파일함

    model.fit(X[train], Y[train], epochs=100, batch_size=5) # X와 Y값으로 모델을 학습하며
    # 각 샘플당 100번 학습하고 한번 학습할 때마다 5개씩 하도록 설정

    k_accuracy = "%.4f" % (model.evaluate(X[test], Y[test])[1]) # X와 Y값으로 모델에 테스트하여 정확성을 도출함
    accuracy.append(k_accuracy)

# 결과 출력
print("\n %.0f fold accuracy : " % n_fold, accuracy)