import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('./경진대회/data/train.csv')
test = pd.read_csv('./경진대회/data/test.csv')
submission = pd.read_csv('./경진대회/data/sample_submission.csv')

train['요일'] = train['요일'].map({'월':0, '화':1, '수':2, '목':3, '금':4})
test['요일'] = test['요일'].map({'월':0, '화':1, '수':2, '목':3, '금':4})

#year = [i[0] for i in train['일자'].str.split('-')]
#month = [i[1] for i in train['일자'].str.split('-')]
#day = [i[2] for i in train['일자'].str.split('-')]

train_year = pd.DataFrame({ '연': [i[0] for i in train['일자'].str.split('-')] })
train_month = pd.DataFrame({ '월': [i[1] for i in train['일자'].str.split('-')] })
train_day = pd.DataFrame({ '일': [i[2] for i in train['일자'].str.split('-')] })

test_year = pd.DataFrame({ '연': [i[0] for i in test['일자'].str.split('-')] })
test_month = pd.DataFrame({ '월': [i[1] for i in test['일자'].str.split('-')] })
test_day = pd.DataFrame({ '일': [i[2] for i in test['일자'].str.split('-')] })

#print("year : ", year)
#print("month : ", month)
#print("day : ", day)


x_train = train[['요일', '본사정원수', '본사출장자수', '본사시간외근무명령서승인건수']]
y1_train = train['중식계']
y2_train = train['석식계']

x_test = test[['요일', '본사정원수', '본사출장자수', '본사시간외근무명령서승인건수']]

# x_train[['연']] = str(train[['일자']])

x_train = pd.concat([x_train, train_year, train_month, train_day], axis=1)
x_test = pd.concat([x_test, test_year, test_month, test_day], axis=1)

print(x_train.head(3))


model1 = RandomForestRegressor(n_jobs=-1, random_state=42)
model2 = RandomForestRegressor(n_jobs=-1, random_state=42)


model1.fit(x_train, y1_train)
model2.fit(x_train, y2_train)


pred1 = model1.predict(x_test)
pred2 = model2.predict(x_test)


submission['중식계'] = pred1
submission['석식계'] = pred2

print("finish")

# submission.to_csv('baseline.csv', index=False)
