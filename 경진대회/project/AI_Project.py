import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# train = pd.read_csv('C:\\Users\\Lee\\Desktop\\AI 응용 프로젝트\\경진대회\\data\\train.csv')
train = pd.read_csv('./경진대회/data/train.csv')
test = pd.read_csv('./경진대회/data/test.csv')
submission = pd.read_csv('./경진대회/data/sample_submission.csv')

# matplotlib 한글깨짐 현상 수정
font_path = "C:\Windows\Fonts\H2GTRM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

print(train.head(10))

# print("----------------------------------------------------------------")

# print(train.info())

# print("----------------------------------------------------------------")

# print(train.describe())

#train['요일'] = train['요일'].map({'월':0, '화':1, '수':2, '목':3, '금':4})
print("----------------------------------------------------------------")
print(train.head(10))

# 데이터 테이블을 그래프로 표현하기
plt.figure(figsize=(6, 4)) # 그래프의 크기를 결정

# heatmap(): 각 항목 간의 상관관계를 나타내 주는 함수.
# 두 항목씩 짝을 지은 뒤 각각 어떤 패턴으로 변화하는지를 관찰하는 함수이다.
# 두 항목이 전혀 다른 패턴으로 변화하고 있으면 0을, 서로 비슷한 패턴으로 변할수록 1에 가까운 값을 출력한다.

sns.heatmap(train.corr(), linewidths=0.1, vmax=0.5, cmap=plt.cm.gist_heat, linecolor='white', annot=True)
# vmax: 색상의 밝기를 조절하는 인자.
# cmap: 미리 정해진 matplotlib 색상의 설정값을 불러오는 인자.

plt.show()