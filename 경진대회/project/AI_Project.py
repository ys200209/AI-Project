import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from matplotlib import font_manager, rc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import re


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

train['요일'] = train['요일'].map({'월':0, '화':1, '수':2, '목':3, '금':4})
test['요일'] = test['요일'].map({'월':0, '화':1, '수':2, '목':3, '금':4})

train = pd.get_dummies(train, columns = ['요일'])
test = pd.get_dummies(test, columns = ['요일'])

print("train = ", train)

x_train = train[['요일_0', '요일_1', '요일_2', '요일_3', '요일_4', '본사정원수', '본사출장자수', '본사시간외근무명령서승인건수', '현본사소속재택근무자수']]
x_train = pd.DataFrame(x_train)
x_test = test[['요일_0', '요일_1', '요일_2', '요일_3', '요일_4', '본사정원수', '본사출장자수', '본사시간외근무명령서승인건수', '현본사소속재택근무자수']]
x_test = pd.DataFrame(x_test)

# 중식 Train 온 핫 인코딩
'''
Y_obj = [i[0] for i in train['중식메뉴']]
print("Y_obj : ", Y_obj)
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
y_encoded_lunch = tf.keras.utils.to_categorical(Y)
x_train = pd.concat([x_train, pd.DataFrame(y_encoded_lunch)], axis=1)
'''

daily_menu = [] # 여기에 Train, Test의 모든 점심, 저녁을 담아서 한번에 처리하도록.

# 중식 Train 온 핫 인코딩
for i in train['중식메뉴']:
    menu = re.sub(r"\([^)]*\)", '', i)
    daily_dinner = menu.split(' ')[0]
    daily_menu.append(daily_dinner)

# 중식 Test 온 핫 인코딩
for i in test['중식메뉴']:
    menu = re.sub(r"\([^)]*\)", '', i)
    daily_lunch = menu.split(' ')[0]
    daily_menu.append(daily_lunch)

# 석식 Train 온 핫 인코딩
for i in train['석식메뉴']:
    menu = re.sub(r"\([^)]*\)", '', i)
    daily_dinner = menu.split(' ')[0]
    daily_menu.append(daily_dinner)

# 석식 Test 온 핫 인코딩
for i in test['석식메뉴']:
    menu = re.sub(r"\([^)]*\)", '', i)
    daily_dinner = menu.split(' ')[0]
    daily_menu.append(daily_dinner)

Y_obj = [i for i in daily_menu]
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
y_encoded_dinner = tf.keras.utils.to_categorical(Y)
x_train = pd.concat([x_train, pd.DataFrame(y_encoded_dinner)], axis=1)
x_test = pd.concat([x_test, pd.DataFrame(y_encoded_dinner)], axis=1)

x_train = x_train.dropna()
x_test = x_test.dropna()

print("x_train : ", x_train)
print("x_test : ", x_test)

y1_train = train['중식계']
y2_train = train['석식계']


model1 = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42, criterion='absolute_error')
model2 = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42, criterion='absolute_error')

# K겹 교차 검증
n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)

accuracy = []

'''
'''

# lunch에 대한 예측값
# x_train_lunch, x_test_lunch, y_train_lunch, y_test_lunch = train_test_split(x_train, y1_train, test_size=0.3, random_state=42)

# dinner에 대한 예측값
# x_train_dinner, x_test_dinner, y_train_dinner, y_test_dinner = train_test_split(x_train, y2_train, test_size=0.3, random_state=42)

model1.fit(x_train, y1_train) # lunch
model2.fit(x_train, y2_train) # dinner 

# print("x_train.info() : ", x_train.info())
# print("x_test.info() : ", x_test.info())



pred1 = model1.predict(x_test)
pred2 = model2.predict(x_test)

# print("lunch = ", model1.score(x_test_lunch, y_test_lunch))
# print("dinner = ", model2.score(x_test_dinner, y_test_dinner))

'''
params = {
    'n_estimators':[100, 150, 175], # 100 [100, 150, 175]
    'max_depth':[2 ,4 ,6, 8, 10, 12], # 2 [2 ,4 ,6, 8, 10, 12]
    'min_samples_leaf':[2 ,4, 6, 8, 12, 18], # 18 [2 ,4, 6, 8, 12, 18]
    'min_samples_split':[4] # 4 [4]
}

# RandomForestClassifier 객체 생성 후 GridSearchCV 수행
# n_jobs = -1 을 지정하면 모든 CPU 코어를 이용해 학습 가능
rf_clf = RandomForestClassifier(n_jobs=-1)      
grid_cv = GridSearchCV(rf_clf, param_grid = params, cv=2, n_jobs=-1)
grid_cv.fit(x_train_lunch, y_train_lunch)

print('최적의 하이퍼 파라미터 :',grid_cv.best_params_)
print('최적의 예측 정확도 :',grid_cv.best_score_)
'''
np.set_printoptions(precision=1)
pred1 = np.round(pred1, 0)
pred2 = np.round(pred2, 0)


submission['중식계'] = pred1
submission['석식계'] = pred2

submission.to_csv('./경진대회/data/baseline.csv', index=False)

print("finish")


