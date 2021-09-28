import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # matplotlib 라이브러리보다 좀 더 정교한 그래프를 그릴게끔 도와주는 seaborn 라이브러리.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf

'''
df = pd.read_csv('./dataset/pima-indians-diabetes.csv',
                names=["pregnant", "plasma", "pressure", "thickness", 
                "insulin", "BMI", "predigree", "age", "class"])
# csv 파일에는 데이터를 설명하는 한 줄의 헤더(header)가 존재하지만 위의 파일에는 헤더가 존재하지 않는다.
# 따라서 이에 names라는 함수를 통해 속성별 키워드(열 이름)를 지정해 주었다.

print(df.head(5)) # 해당 csv 파일의 첫 다섯 줄을 불러온다.

print("----------------------------------------------------------------")

print(df.info()) # 데이터의 전반적인 정보를 확인하는 .info() 함수

print("----------------------------------------------------------------")

print(df[['pregnant', 'class']].groupby(['pregnant'], 
            as_index=False).mean().sort_values(by="pregnant", ascending=True))
# 임신 횟수와 당뇨병 발병 확률의 상관관계
# 1. groupby() 함수를 사용해 임신 횟수 정보를 기준으로 하는 그룹을 생성.
# 2. as_index=False 를 통해 pregnant 정보 옆에 새로운 인덱스를 생성.
# 3. mean() 함수를 사용해 평균을 구하고 임신 횟수에 대한 오름차순 정렬로 정보를 가공한다.


# 데이터 테이블을 그래프로 표현하기
plt.figure(figsize=(12, 12)) # 그래프의 크기를 결정

# heatmap(): 각 항목 간의 상관관계를 나타내 주는 함수.
# 두 항목씩 짝을 지은 뒤 각각 어떤 패턴으로 변화하는지를 관찰하는 함수이다.
# 두 항목이 전혀 다른 패턴으로 변화하고 있으면 0을, 서로 비슷한 패턴으로 변할수록 1에 가까운 값을 출력한다.

sns.heatmap(df.corr(), linewidths=0.1, vmax=0.5, cmap=plt.cm.gist_heat, linecolor='white', annot=True)
# vmax: 색상의 밝기를 조절하는 인자.
# cmap: 미리 정해진 matplotlib 색상의 설정값을 불러오는 인자.

plt.show()
# 이 결과로 plasma(공복 혈당 농도) 항목이 당뇨병 발병에 가장 관련이 있는 것으로 나타났다.

# 따라서 plasma 항목과 당뇨병 발병 항목만 따로 떼어 두 항목 간의 관계를 살펴보면
grid = sns.FacetGrid(df, col="class")
grid.map(plt.hist, "plasma", bins=10)
plt.show()
# 이 결과로 당뇨병이 발병한 사람의 공복 혈당 농도는 그렇지 않은 사람보다 150~200 의 수치가 매우 높았다.
'''

# seed 값 생성
np.random.seed(3)
tf.random.set_seed(3)

# 데이터 로드
dataset = np.loadtxt('./dataset/pima-indians-diabetes.csv',
    delimiter=",")

X = dataset[:, 0:8]
Y = dataset[:, 8]

# 모델의 설정
model = Sequential()
model.add(Dense(12, input_dim=8, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# 모델 컴파일
model.compile(loss="binary_crossentropy",
                optimizer="adam",
                metrics=["accuracy"])

# 모델 실행
model.fit(X, Y, epochs=200, batch_size=10)

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))

