from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  


train = pd.read_csv('./경진대회/data/train.csv')
test = pd.read_csv('./경진대회/data/test.csv')
submission = pd.read_csv('./경진대회/data/sample_submission.csv')

train = pd.get_dummies(train, columns = ['요일'])
test = pd.get_dummies(test, columns = ['요일'])

X_train = train[['요일_월', '요일_화', '요일_수', '요일_목', '요일_금', '본사정원수', '본사출장자수', '본사시간외근무명령서승인건수', '현본사소속재택근무자수']]
y1_train = train['중식계']
y2_train = train['석식계']
x_test = test[['요일_월', '요일_화', '요일_수', '요일_목', '요일_금', '본사정원수', '본사출장자수', '본사시간외근무명령서승인건수', '현본사소속재택근무자수']]

x_train, x_test, y_train, y_test = train_test_split(X_train, y1_train, test_size=0.3, random_state=42)

# rf = RandomForestClassifier(random_state=0)
rf = RandomForestRegressor(n_estimators = 239, n_jobs=-1, random_state=42, criterion='mae')
rf.fit(x_train,y_train)
pred = rf.predict(x_test)
accuracy = accuracy_score(x_test, y_train)

print("랜덤 포레스트 정확도 : ", accuracy)


'''
submission['중식계'] = pred1
submission['석식계'] = pred2

submission.to_csv('./경진대회/data/baseline.csv', index=False)
'''
