import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from category_encoders import OneHotEncoder


train = pd.read_csv('./경진대회/data/train.csv')
test = pd.read_csv('./경진대회/data/test.csv')
submission = pd.read_csv('./경진대회/data/sample_submission.csv')

# 코로나 전후 상황 비교
'''
before = train[(train['일자'] >= '2019-01-01')&(train['일자'] <= '2019-12-31')][['일자', '중식계', '석식계']]
after = train[train['일자'] >= '2020-01-01'][['일자', '중식계', '석식계']]
print('점심 :', '2019년도 : ', round(before.중식계.mean(), 2), ', 2020년도 : ', round(after.중식계.mean(), 2))
print('저녁 :', '2019년도 : ', round(before.석식계.mean(), 2), ', 2020년도 : ', round(after.석식계.mean(), 2))
print()
'''

print("석식이 없는 행을 삭제하기 전 행의 수 : ", len(train))
train_delete_row = train[train['석식계'] == 0.0].index+1
train = train.drop(train_delete_row)
print("검열된 삭제 행 수 : ", train_delete_row.size)
print("삭제를 마친 뒤 행의 수 : ", len(train))
train = train.reset_index(drop=True)


# 메뉴에서 불필요한 문자 제거
lunch = []
dinner = []
train_lunch_rice = [] # 점심밥의 메뉴
train_lunch_soup = [] # 점심국의 메뉴
train_lunch_main = [] # 점심 메인 메뉴
train_dinner_rice = [] # 저녁밥의 메뉴
train_dinner_soup = [] # 저녁국의 메뉴
train_dinner_main = [] # 저녁 메인 메뉴

# Train 점심 메뉴
for day in range(len(train)):
    tmp = train.loc[day,'중식메뉴'].split(' ') # 스페이스로 구분
    tmp = ' '.join(tmp).split()    # 빈칸 제거

    for menu in tmp:
        if '(' in menu:
            tmp.remove(menu)
    lunch.append(tmp) 


for menu in lunch: # 여기까지 train 파일 중식 가져오기
    if '쌀밥' in menu[0]:
        train_lunch_rice.append('쌀밥')
    else :
        train_lunch_rice.append(menu[0])
    train_lunch_soup.append(menu[1])
    train_lunch_main.append(menu[2])



# Train 저녁 메뉴
for day in range(len(train)):
    tmp = train.loc[day,'석식메뉴'].split(' ') # 스페이스로 구분
    tmp = ' '.join(tmp).split()    # 빈칸 제거

    for menu in tmp:
        if '(' in menu:
            tmp.remove(menu)
    dinner.append(tmp) 

'''print("dinner.len : ", len(dinner))
print("dinner[0].len : ", len(dinner[0]))
print("dinner : ", dinner)'''

for menu in dinner: # 여기까지 train 파일 석식 가져오기
    if len(menu) == 0:
        train_dinner_rice.append('None')
        train_dinner_soup.append('None')
        train_dinner_main.append('None')
    elif '*' in menu: # (204번 인덱스에서 메뉴명이 '*' 인 행 발견)
        train_dinner_rice.append('None')
        train_dinner_soup.append('None')
        train_dinner_main.append('None')
    elif '자기계발의날' in menu: # (315번 인덱스에서 메뉴명이 '자기계발의날' 인 행 발견)
        train_dinner_rice.append('None')
        train_dinner_soup.append('None')
        train_dinner_main.append('None')
    elif '*자기계발의날*' in menu: # (339번 인덱스에서 메뉴명이 '*자기계발의날*' 인 행 발견)
        train_dinner_rice.append('None')
        train_dinner_soup.append('None')
        train_dinner_main.append('None')
    elif '가정의날' in menu: # (358번 인덱스에서 메뉴명이 '가정의날' 인 행 발견)
        train_dinner_rice.append('None')
        train_dinner_soup.append('None')
        train_dinner_main.append('None')
    elif '자기개발의날' in menu: # (702번 인덱스에서 메뉴명이 '자기개발의날' 인 행 발견)
        train_dinner_rice.append('None')
        train_dinner_soup.append('None')
        train_dinner_main.append('None')
    else:
        train_dinner_rice.append(menu[0])
        train_dinner_soup.append(menu[1])
        train_dinner_main.append(menu[2])
    


lunch = []
dinner = []
test_lunch_rice = []
test_lunch_soup = []
test_lunch_main = []
test_dinner_rice = []
test_dinner_soup = []
test_dinner_main = []

# Test 점심 메뉴
for day in range(len(test)):
    tmp = test.loc[day,'중식메뉴'].split(' ') # 스페이스로 구분
    tmp = ' '.join(tmp).split()    # 빈칸 제거

    for menu in tmp:
        if '(' in menu:
            tmp.remove(menu)
    lunch.append(tmp) 

for menu in lunch: # 여기까지 test 파일 중식 가져오기
    test_lunch_rice.append(menu[0])
    test_lunch_soup.append(menu[1])
    test_lunch_main.append(menu[2])

# Test 저녁 메뉴
for day in range(len(test)):
    tmp = test.loc[day,'석식메뉴'].split(' ') # 스페이스로 구분
    tmp = ' '.join(tmp).split()    # 빈칸 제거

    for menu in tmp:
        if '(' in menu:
            tmp.remove(menu)
    dinner.append(tmp) 

for menu in dinner: # 여기까지 test 파일 석식 가져오기
    if '쌀밥' in menu[0]:
        test_dinner_rice.append('쌀밥')
    else :
        test_dinner_rice.append(menu[0])
    test_dinner_soup.append(menu[1])
    test_dinner_main.append(menu[2])

'''
train_lunch_rice = pd.DataFrame({ 'lunch_rice': train_lunch_rice })
train_lunch_main = pd.DataFrame({ 'lunch_main': train_lunch_main })
test_lunch_rice = pd.DataFrame({ 'lunch_rice': test_lunch_rice })
test_lunch_main = pd.DataFrame({ 'lunch_main': train_lunch_main })
'''

train['요일'] = train['요일'].map({'월':0, '화':1, '수':2, '목':3, '금':4})
# train['요일'] = train['요일'].dropna()
test['요일'] = test['요일'].map({'월':0, '화':1, '수':2, '목':3, '금':4})
# test['요일'] = test['요일'].dropna()

#year = [i[0] for i in train['일자'].str.split('-')]
#month = [i[1] for i in train['일자'].str.split('-')]
#day = [i[2] for i in train['일자'].str.split('-')]


train_year = pd.DataFrame({ '연': [i[0] for i in train['일자'].str.split('-')] })
train_month = pd.DataFrame({ '월': [i[1] for i in train['일자'].str.split('-')] })
train_day = pd.DataFrame({ '일': [i[2] for i in train['일자'].str.split('-')] })

test_year = pd.DataFrame({ '연': [i[0] for i in test['일자'].str.split('-')] })
test_month = pd.DataFrame({ '월': [i[1] for i in test['일자'].str.split('-')] })
test_day = pd.DataFrame({ '일': [i[2] for i in test['일자'].str.split('-')] })

# 근무인원 피쳐추가 : 본사정원수 - ( 본사출장자수 + 본사휴가자수 + 현본사소속재택근무자수 )
train_work_number = pd.DataFrame({ '근무인원': (train['본사정원수'] - ( train['본사휴가자수']+ train['본사출장자수'] + train['현본사소속재택근무자수'] ))})
test_work_number = pd.DataFrame({ '근무인원': (test['본사정원수'] - ( test['본사휴가자수'] + test['본사출장자수'] + test['현본사소속재택근무자수'] ))})





# x_train = train[['요일']]
y1_train = train['중식계']
y2_train = train['석식계']

# print("결측치 확인 1 : \n", x_train.isna().any())
# print("len(x_train) : ", len(x_train))

# 전처리로 생성된 피쳐 : 요일, 연, 월, 일, 실근무인원
# x_train = pd.concat([x_train, train_year, train_month, train_day, train_work_number, lunch_rice, lunch_main], axis=1)

# x_train = [[train['요일'], train_year, train_month, train_day, train_work_number, lunch_rice, lunch_main]]
# x_test = [[test['요일'], test_year, test_month, test_day, test_work_number, lunch_rice, lunch_main]]
x_train = pd.concat([train['요일'], train_year, train_month, train_day, train_work_number, train['본사시간외근무명령서승인건수']], axis=1, ignore_index = True)
x_test = pd.concat([test['요일'], test_year, test_month, test_day, test_work_number, test['본사시간외근무명령서승인건수']], axis=1, ignore_index = True)
x_train.columns = ['요일', '연', '월', '일', '근무인원', '야근수']
x_test.columns = ['요일', '연', '월', '일', '근무인원', '야근수']

print("len(train_lunch_rice) : ", len(train_lunch_rice))
print("len(x_train) : ", len(x_train))

# x_train = x_train.astype({'근무인원' : 'int'})
x_train['lunch_rice'] = train_lunch_rice
x_train['lunch_soup'] = train_lunch_soup
x_train['lunch_main'] = train_lunch_main
x_train['dinner_rice'] = train_dinner_rice
x_train['dinner_soup'] = train_dinner_soup
x_train['dinner_main'] = train_dinner_main

# x_test = x_test.astype({'근무인원' : 'int'})
x_test['lunch_rice'] = test_lunch_rice
x_test['lunch_soup'] = test_lunch_soup
x_test['lunch_main'] = test_lunch_main
x_test['dinner_rice'] = test_dinner_rice
x_test['dinner_soup'] = test_dinner_soup
x_test['dinner_main'] = test_dinner_main


print("x_train.head(2) : ", x_train.head(2))
print(x_train['lunch_rice'].value_counts())
# x_train.dropna(axis=0)

# x_train = x_train.dropna()
# x_train = x_train.reset_index(drop=True)


'''
i = 0
for count in x_train['근무인원']:
    print(i, " : ", count, " type : ", type(count))
    if count == np.NaN:
        print("nan...")
    i+=1
'''


'''
# 밥 개수 확인
bob_df = pd.DataFrame(x_train['lunch_rice'].value_counts().reset_index())
# print("count : \n ", bob_df.head(10))

# 반찬 데이터 개수 확인
banchan_list = []
tmp = x_train['lunch_main']

for j in range(1162):
    banchan_list.append(tmp[j])

banchan_df = pd.DataFrame(pd.DataFrame(banchan_list).value_counts())
banchan_df.columns = ['banchan']
banchan_df.reset_index(inplace = True)
banchan_df.columns = ['index', 'banchan']
print(banchan_df.head(10))
print("len(banchan_df) : ", len(banchan_df))

print("석식이 없는 행을 삭제하기 전 행의 수 : ", len(x_train))
train_delete_row = x_train[x_train['석식계'] == 0.0].index
x_train = x_train.drop(train_delete_row)
print("검열된 삭제 행 수 : ", train_delete_row.size)
print("삭제를 마친 뒤 행의 수 : ", len(x_train))
'''

params = {
    'min_samples_leaf' :[10,12,15],
    'n_estimators' : [200,300,450,600],
    'max_depth' : [1, 5, 10, 20],
    'max_features' : [ 0.2, 0.5, 0.8, 1]
}

lunch_r = RandomForestRegressor()
dinner_r = RandomForestRegressor()

lunch_model = RandomizedSearchCV(lunch_r, params, scoring='neg_mean_absolute_error')
dinner_model = RandomizedSearchCV(dinner_r, params, scoring='neg_mean_absolute_error')


'''
print("x_train.loc(0) : ", x_train.loc[[0]])
print("x_train.loc(1000) : ", x_train.loc[[1000]])
print("x_train.loc(1161) : ", x_train.loc[1161])
print("x_train.loc(1162) : ", x_train.loc[1162])
print("x_train.loc(1204) : ", x_train.loc[1204])
'''

# 메뉴 원핫 인코딩
# encoder = OneHotEncoder(use_cat_names = True, cols = [6, 7],)
# x_train = encoder.fit_transform(x_train)
# x_test = encoder.fit_transform(x_test)

# Train 요일 인코딩
x_train['요일'] =  x_train['요일'].astype('category')
x_train['요일'] = x_train.요일.cat.codes
# Train 점심밥 인코딩
x_train['lunch_rice'] = x_train['lunch_rice'].astype('category')
x_train['lunch_rice'] = x_train.lunch_rice.cat.codes
# Train 점심국 인코딩
x_train['lunch_soup'] = x_train['lunch_soup'].astype('category')
x_train['lunch_soup'] = x_train.lunch_soup.cat.codes
# Train 점심반찬 인코딩
x_train['lunch_main'] = x_train['lunch_main'].astype('category')
x_train['lunch_main'] = x_train.lunch_main.cat.codes
# Train 저녁밥 인코딩
x_train['dinner_rice'] = x_train['dinner_rice'].astype('category')
x_train['dinner_rice'] = x_train.dinner_rice.cat.codes
# Train 저녁국 인코딩
x_train['dinner_soup'] = x_train['dinner_soup'].astype('category')
x_train['dinner_soup'] = x_train.dinner_soup.cat.codes
# Train 저녁반찬 인코딩
x_train['dinner_main'] = x_train['dinner_main'].astype('category')
x_train['dinner_main'] = x_train.dinner_main.cat.codes
# Test 요일 인코딩
x_test['요일'] =  x_test['요일'].astype('category')
x_test['요일'] = x_test.요일.cat.codes
# Test 점심밥 인코딩
x_test['lunch_rice'] = x_test['lunch_rice'].astype('category')
x_test['lunch_rice'] = x_test.lunch_rice.cat.codes
# Test 점심국 인코딩
x_test['lunch_soup'] = x_test['lunch_soup'].astype('category')
x_test['lunch_soup'] = x_test.lunch_soup.cat.codes
# Test 점심반찬 인코딩
x_test['lunch_main'] = x_test['lunch_main'].astype('category')
x_test['lunch_main'] = x_test.lunch_main.cat.codes
# Test 저녁밥 인코딩
x_test['dinner_rice'] = x_test['dinner_rice'].astype('category')
x_test['dinner_rice'] = x_test.dinner_rice.cat.codes
# Test 저녁밥 인코딩
x_test['dinner_soup'] = x_test['dinner_soup'].astype('category')
x_test['dinner_soup'] = x_test.dinner_soup.cat.codes
# Test 저녁반찬 인코딩
x_test['dinner_main'] = x_test['dinner_main'].astype('category')
x_test['dinner_main'] = x_test.dinner_main.cat.codes

print("x_train.head() : ", x_train.head())
print("x_test.head() : ", x_test.head())

print("x_train.info() : ", x_train.info())
print("x_test.info() : ", x_test.info())
#print("np.isnan(x_train) : ", np.isnan(x_train))
#print("np.isnan(x_test) : ", np.isnan(x_test))



# 훈련
lunch_model.fit(x_train, y1_train)
dinner_model.fit(x_train, y2_train)

lunch_best = lunch_model.best_score_
dinner_best = dinner_model.best_score_


print('점심 베이스라인 모델 에러값(mae) : ',lunch_best)
print("점심 최적의 하이퍼 파라미터값 : ", lunch_model.best_estimator_)

print('저녁 베이스라인 모델 에러값(mae) : ', dinner_best)
print("저녁 최적의 하이퍼 파라미터값 : ", dinner_model.best_estimator_)

pred1 = lunch_model.predict(x_test)
pred2 = dinner_model.predict(x_test)




'''
model1 = RandomForestRegressor(n_jobs=-1, random_state=42, max_depth=10, max_features=0.8, min_samples_leaf=10, n_estimators=600)
model2 = RandomForestRegressor(n_jobs=-1, random_state=42, max_depth=10, max_features=0.8, min_samples_leaf=10, n_estimators=600)


model1.fit(x_train, y1_train)
model2.fit(x_train, y2_train)

pred1 = model1.predict(x_test)
pred2 = model2.predict(x_test)
'''
submission['중식계'] = pred1
submission['석식계'] = pred2



submission.to_csv('./경진대회/data/baseline.csv', index=False)

print("finish")