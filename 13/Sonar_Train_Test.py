from keras.models import Sequential, load_model
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(3)
df = pd.read_csv('./dataset/sonar.csv', header=None)

dataset = df.values
X = dataset[:, 0:60].astype(float) # 0~60개의 피쳐를 X 변수에 할당
Y_obj = dataset[:, 60] # 결과값을 Y_obj 변수에 할당

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj) # 결과값인 문자열을 인코딩하여 Y 변수에 할당

# 학습셋과 테스트셋을 나눔
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
test_size=0.3, random_state=seed)
# X와 Y값 데이터 중에서 30%를 X_test, Y_test로 별도 분리하여 트레인값과 테스트값이 겹치지않게 모델을 학습시켜
# 정확성에 대한 신뢰도를 높이는 방법

model = Sequential()
model.add(Dense(24, input_dim=60, activation="relu")) # 60개의 input과 24개의 output을 가지는 입력층을 생성
model.add(Dense(10, activation="relu")) # 10개의 input과 1개의 output을 가지는 은닉층을 생성
model.add(Dense(1, activation="sigmoid")) # 1개의 input을 가지는 출력층을 생성 (활성화 함수는 sigmoid)

model.compile(loss="mean_squared_error", # [평균 제곱 오차] 오차 함수를 이용.
    optimizer="adam", # 최적화 함수는 adam으로 설정.
    metrics=["accuracy"]) # 결과를 accuracy로 출력되도록.

model.fit(X_train, Y_train, epochs=130, batch_size=5) 
# 5개의 샘플을 비교하며 값을 수정해나가며 총 26번의 갱신이 일어남
model.save("my_model.h5") # 모델을 컴퓨터에 저장

del model # 테스트를 위해 메모리 내의 모델을 삭제
model = load_model("my_model.h5") # 모델을 새로 불러옴

print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))
# 불러온 모델로 테스트 실행

