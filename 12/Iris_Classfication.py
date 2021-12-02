from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_data_flow_ops import tensor_array_split_eager_fallback

# seed 값 설정
np.random.seed(3)
tf.random.set_seed(3)
# 랜덤이지만 똑같은 랜덤 값을 출력하도록 하기 위해서 시드값을 설정한다.

# 데이터 입력
df = pd.read_csv("./dataset/iris.csv", names = ["sepal_length",
    "sepal_width", "petal_length", "petal_width", "species"])

# 그래프로 확인
sns.pairplot(df, hue="species")
plt.show()

# 데이터 분류
dataset = df.values
X = dataset[:, 0:4].astype(float) # dataset의 0~3 인덱스인 피쳐값들의 타입을 float형식으로 변환해준다.
Y_obj = dataset[:, 4]

# 문자열을 숫자로 변환
e = LabelEncoder()
e.fit(Y_obj) # 결과값인 문자열을 라벨 인코딩을 이용하여 변환
Y = e.transform(Y_obj)
Y_encoded = tf.keras.utils.to_categorical(Y) # 문자열 결과값을 카테고리 형식으로 변환하여 Y_encoded 변수에 저장

# 모델의 설정
model = Sequential()
model.add(Dense(16, input_dim=4, activation="relu")) # 4개의 input과 16개의 output을 가지는 입력층 생성
model.add(Dense(3, activation="softmax")) # 3개의 input을 가지는 출력층 생성. 활성화 함수는 softmax를 사용하였다.

# 모델 컴파일
model.compile(loss="categorical_crossentropy",  # 오차 함수를 categorical_crossentropy를 사용하고
            optimizer="adam", # 최적화 함수를 adam 함수로 설정하고
            metrics=["accuracy"]) # 결과 출력 방식을 accuracy로 설정하였다.

# 모델 실행
model.fit(X, Y_encoded, epochs=50, batch_size=1) # X값과 카테고리화 된 Y값을 이용해 학습하며 
# 각 샘플을 50번 반복 학습하고 한번의 학습에 하나의 샘플씩 학습하도록 설정해주었다.

# 결과 출력
print("\n Accuracy : %.4f" % (model.evaluate(X, Y_encoded)[1]))