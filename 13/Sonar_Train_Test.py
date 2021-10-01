from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(seed)
df = pd.read_csv('./dataset/sonar.csv', header=None)

dataset = df.values
X = dataset[:, 0:60].astype(float)
Y_obj = dataset[:, 60]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

print("Y = ", Y)
'''
# 학습셋과 테스트셋의 구분
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(24, input_dim=60, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="mean_squared_error", # [최소 제곱 오차] 활성화 함수를 이용
    optimizer="adam",
    metrics=["accuracy"])

model.fit(X_train, Y_train, epochs=130, batch_size=5) # 5개의 샘플을 비교하며 값을 수정해나가며 총 26번의 갱신이 일어남

# 테스트셋에 모델 적용
print("\n Test Accuracy: %.04f" % (model.evaluate(X_test, Y_test)[1]))

model.save('my_model.h5')
'''
