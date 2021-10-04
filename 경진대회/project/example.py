from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


train = pd.read_csv('./경진대회/data/train.csv')
test = pd.read_csv('./경진대회/data/test.csv')
submission = pd.read_csv('./경진대회/data/sample_submission.csv')
train['요일'] = train['요일'].map({'월':0, '화':1, '수':2, '목':3, '금':4})
train = pd.get_dummies(train, columns = ['요일'])


print("train 1 = ", train)
x_train = train[['요일_0', '요일_1', '요일_2', '요일_3', '요일_4', '본사정원수', '본사출장자수', '본사시간외근무명령서승인건수', '현본사소속재택근무자수', '중식메뉴']]
lunch = pd.get_dummies(x_train['중식메뉴'], columns = ['중식메뉴'])

Y_obj = train['중식메뉴']
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
y_encoded = tf.keras.utils.to_categorical(Y)

print("y_encoded : ", y_encoded)

x_train = x_train.append(lunch)
dinner = pd.get_dummies(train['석식메뉴'])

print("x_train 2 = ", x_train)
print(x_train['본사정원수'])
lunch.to_csv('./경진대회/data/lunch.csv', index=False)
dinner.to_csv('./경진대회/data/dinner.csv', index=False)

'''
test = pd.get_dummies(test, columns = ['요일'])

X_train = train[['요일_월', '요일_화', '요일_수', '요일_목', '요일_금', '본사정원수', '본사출장자수', '본사시간외근무명령서승인건수', '현본사소속재택근무자수']]
X_train = pd.concat([X_train, lunch], axis=1)
# print("X_train = \n", X_train)
y1_train = train['중식계']
y2_train = train['석식계']
x_test = test[['요일_월', '요일_화', '요일_수', '요일_목', '요일_금', '본사정원수', '본사출장자수', '본사시간외근무명령서승인건수', '현본사소속재택근무자수']]
x_test = pd.concat([x_test, lunch], axis=1)

x_train, x_test, y_train, y_test = train_test_split(X_train, y1_train, test_size=0.3, random_state=42)

# rf = RandomForestClassifier(random_state=0)
rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(x_train, y_train)
pred = rf_clf.predict(x_test)
accuracy = accuracy_score(y_test, pred)
print('랜덤 포레스트 정확도: {:.4f}'.format(accuracy))
'''

'''
submission['중식계'] = pred1
submission['석식계'] = pred2

submission.to_csv('./경진대회/data/baseline.csv', index=False)
'''
