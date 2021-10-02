import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from sklearn.preprocessing import LabelEncoder
import re

regex = "\(.*\)|\s-\s.*"
# regex = "\(.*\)"
# regex = "\s+"
# regex = "|\s-\s.*"


train = pd.read_csv('./경진대회/data/train.csv')
test = pd.read_csv('./경진대회/data/test.csv')
submission = pd.read_csv('./경진대회/data/sample_submission.csv')



# train['요일'] = train['요일'].map({'월':0, '화':1, '수':2, '목':3, '금':4})
print(pd.get_dummies(train, columns = ['요일']))

train = pd.get_dummies(train, columns = ['요일'])
# test['요일'] = test['요일'].map({'월':0, '화':1, '수':2, '목':3, '금':4})

test = pd.get_dummies(test, columns = ['요일'])
print(train.info())
print(train.head(10))

x_train = train[['요일_월', '요일_화', '요일_수', '요일_목', '요일_금', '본사정원수', '본사출장자수', '본사시간외근무명령서승인건수', '현본사소속재택근무자수']]
y1_train = train['중식계']
y2_train = train['석식계']

x_test = test[['요일_월', '요일_화', '요일_수', '요일_목', '요일_금', '본사정원수', '본사출장자수', '본사시간외근무명령서승인건수', '현본사소속재택근무자수']]


model1 = RandomForestRegressor(n_jobs=-1, random_state=42)
model2 = RandomForestRegressor(n_jobs=-1, random_state=42)

model1.fit(x_train, y1_train)
model2.fit(x_train, y2_train)

pred1 = model1.predict(x_test)
pred2 = model2.predict(x_test)

pred1 = np.round(model1.predict(x_test), 1)
pred2 = np.round(model2.predict(x_test), 1)

submission['중식계'] = pred1
submission['석식계'] = pred2

submission.to_csv('./경진대회/data/ttttttttttest.csv', index=False)

# x_train['요일'] = pd.get_dummies(x_train['요일'])



# ttttttttttest = pd.read_csv('./경진대회/data/ttttttttttest.csv')

'''
x1 = [i[0] for i in x_train]
x2 = [i[1] for i in x_train]
x3 = [i[2] for i in x_train]
x4 = [i[3] for i in x_train]
x5 = [i[4] for i in x_train]

x1_data = np.array(x1)
x2_data = np.array(x2)
x3_data = np.array(x3)
x4_data = np.array(x4)
x5_data = np.array(x5)
y1_data = np.array(y1_train)
y2_data = np.array(y2_train)
'''

w1 = 0
w2 = 0
w3 = 0
w4 = 0
w5 = 0
b = 0

epochs = 2001

'''
le = LabelEncoder()
le = le.fit(train['중식메뉴'])   #train['col']을 fit
train['중식메뉴'] = le.transform(train['중식메뉴'])   #train['col']에 따라 encoding
# test['col'] = le.transform(test['col'])   #train['col']에 따라 encoding
'''

print("중식메뉴 : ")

lunch = []
daily_lunch = []


# 일일 메뉴에서 ( 원산지 )와 공백을 제거한 순수 메뉴 리스트
'''
for i in range(3):
    menu = train['중식메뉴'][i]
    
    menu = re.sub(r"\([^)]*\)", '', menu)
    daily_lunch = menu.split()

    lunch.append(daily_lunch)
'''

for i in train['중식메뉴']:
    
    menu = re.sub(r"\([^)]*\)", '', i)
    daily_lunch = menu.split()
    lunch.append(daily_lunch)

# print("lunch = ", lunch)

'''
print("lunch.size = ", len(lunch[0]))
print("side_menu1.size = ", len(lunch[1]))
print("side_menu2.size = ", len(lunch[2]))
print("side_menu3.size = ", len(lunch[3]))
print("side_menu4.size = ", len(lunch[4]))
print("side_menu5.size = ", len(lunch[5]))
print("side_menu6.size = ", len(lunch[6]))


rice = [i[0] for i in lunch]
# print("rice... ===== ", rice)

for i in lunch:
    print("side_menu5.size = ", len(lunch[5]))


side_menu1 = [i[1] for i in lunch]
side_menu2 = [i[2] for i in lunch]
side_menu3 = [i[3] for i in lunch]
side_menu4 = [i[4] for i in lunch]
# side_menu5 = [i[5] for i in lunch]
side_menu6 = [i[6] for i in lunch]
'''

'''
print("rice... : ", rice)
print("side_menu1... : ", side_menu1)
print("side_menu2... : ", side_menu2)
print("side_menu3... : ", side_menu3)
print("side_menu4... : ", side_menu4)
print("side_menu5... : ", side_menu5)
print("side_menu6... : ", side_menu6)


e = LabelEncoder()
e = e.fit(rice)
Y = e.transform(rice)
print("Encoding : rice = ", Y)

e = e.fit(side_menu1)
side_menu1 = e.transform(side_menu1)
#print("Encoding : side_menu1 = ", Y)

e = e.fit(side_menu2)
Y = e.transform(side_menu2)
#print("Encoding : side_menu2 = ", Y)

e = e.fit(side_menu3)
Y = e.transform(side_menu3)
#print("Encoding : side_menu3 = ", Y)

e = e.fit(side_menu4)
Y = e.transform(side_menu4)
#print("Encoding : side_menu4 = ", Y)

e = e.fit(side_menu5)
Y = e.transform(side_menu5)
#print("Encoding : side_menu5 = ", Y)

e = e.fit(side_menu6)
Y = e.transform(side_menu6)
'''
#print("Encoding : side_menu6 = ", Y)

# 이제 구별은 할 수 있게 되었지만, 관건은 0~n까지의 수로 구분되어 있는 이 데이터를
# '숫자의 크기' 로 평가하는것이 아닌, 음식을 '구분' 하는 용도로 사용할 수 있는가.. 이다.

# '요일' 을 숫자로 매핑하지 말고 '원 핫 인코딩' 기법으로 변환 시킨 뒤 학습



#menu = train['중식메뉴'][0]
# menu = text_to_word_sequence(menu)
# print("menu2 = ", menu)
'''
dataFrame_lunch = pd.DataFrame({
    'rice':rice,
    'side_menu1':side_menu1,
    'side_menu2':side_menu2,
    'side_menu3':side_menu3,
    'side_menu4':side_menu4,
    'side_menu5':side_menu5,
    'side_menu6':side_menu6
})
'''
dataFrame_lunch.to_csv('./경진대회/data/lunch.csv', index=False)