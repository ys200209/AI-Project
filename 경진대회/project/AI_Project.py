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
print(train.info())
print("----------------------------------------------------------------")
print(test.info())
print("----------------------------------------------------------------")
print(submission.info())

# 데이터 테이블을 그래프로 표현하기
# plt.figure(figsize=(10, 6)) # 그래프의 크기를 결정

# heatmap(): 각 항목 간의 상관관계를 나타내 주는 함수.
# 두 항목씩 짝을 지은 뒤 각각 어떤 패턴으로 변화하는지를 관찰하는 함수이다.
# 두 항목이 전혀 다른 패턴으로 변화하고 있으면 0을, 서로 비슷한 패턴으로 변할수록 1에 가까운 값을 출력한다.

# sns.heatmap(train.corr(), linewidths=0.1, vmax=0.5, cmap=plt.cm.gist_heat, linecolor='white', annot=True)
# vmax: 색상의 밝기를 조절하는 인자.
# cmap: 미리 정해진 matplotlib 색상의 설정값을 불러오는 인자.

# plt.show()

'''model = RandomForestRegressor(n_jobs=-1, random_state=42)
model.fit'''

# train['요일'] = train['요일'].map({'월':0, '화':1, '수':2, '목':3, '금':4})
# test['요일'] = test['요일'].map({'월':0, '화':1, '수':2, '목':3, '금':4})

train = pd.get_dummies(train, columns = ['요일'])
test = pd.get_dummies(test, columns = ['요일'])

x_train = train[['요일_월', '요일_화', '요일_수', '요일_목', '요일_금', '본사정원수', '본사출장자수', '본사시간외근무명령서승인건수', '현본사소속재택근무자수']]
y1_train = train['중식계']
y2_train = train['석식계']
x_test = test[['요일_월', '요일_화', '요일_수', '요일_목', '요일_금', '본사정원수', '본사출장자수', '본사시간외근무명령서승인건수', '현본사소속재택근무자수']]

model_score1 = 0
model_score1_index = 0
model_score2 = 0
model_score2_index = 0

'''
for i in range(230, 330):
    print("i = ", i)
    model1 = RandomForestRegressor(n_estimators = i, n_jobs=-1, random_state=42)
    model2 = RandomForestRegressor(n_estimators = i, n_jobs=-1, random_state=42)

    model1.fit(x_train, y1_train)
    model2.fit(x_train, y2_train)

    if (model_score1 < model1.score(x_train, y1_train)):
        model_score1 = model1.score(x_train, y1_train)
        model_score1_index = i

    if (model_score2 < model2.score(x_train, y2_train)):
        model_score2 = model2.score(x_train, y2_train)
        model_score2_index = i
'''
    
print("모델1 트리 수 : ", model_score1_index)
print('결정계수1 : ', model_score1)
print("모델2 트리 수 : ", model_score2_index)
print('결정계수2 : ', model_score2)

model1 = RandomForestRegressor(n_estimators = 239, n_jobs=-1, random_state=42, criterion='mae') # 파라미터 여부 n_estimators = 239
model2 = RandomForestRegressor(n_estimators = 252, n_jobs=-1, random_state=42, criterion='mae') # 파라미터 여부 n_estimators = 252

model1.fit(x_train, y1_train)
model2.fit(x_train, y2_train)

print("model1.feature_importances_ : ", model1.feature_importances_)
print("model2.feature_importances_ : ", model2.feature_importances_)

pred1 = model1.predict(x_test)
pred2 = model2.predict(x_test)

print("중식계 pred1 = ", pred1)
print("석식계 pred2 = ", pred2)
print("\n------------------------------------------------------------\n")


'''
model1 = RandomForestRegressor(n_estimators = i, n_jobs=-1, random_state=42)
model2 = RandomForestRegressor(n_estimators = i, n_jobs=-1, random_state=42)

model1.fit(x_train, y1_train)
model2.fit(x_train, y2_train)


relation_square1 = model1.score(x_train, y1_train)
relation_square2 = model2.score(x_train, y2_train)
print('결정계수1 : ', relation_square1)
print('결정계수2 : ', relation_square2)
'''

np.set_printoptions(precision=1)
pred1 = np.round(model1.predict(x_test), 0)
pred2 = np.round(model2.predict(x_test), 0)



# 현재값을 그래프로 나타내보기


submission['중식계'] = pred1
submission['석식계'] = pred2

submission.to_csv('./경진대회/data/baseline.csv', index=False)