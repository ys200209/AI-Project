import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


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

count = 0
for i in train['석식계']:
    if i == 0.0:
        count+=1
print("아직도 0인 석식계 : ", count)
'''
    if count >= 10:
        break
    count+=1
'''
# 메뉴에서 불필요한 문자 제거
lunch = []
for day in range(len(train)):
    tmp = train.iloc[day, 8].split(' ') # 공백으로 문자열 구분
    tmp = ' '.join(tmp).split()    # 빈 원소 삭제

    search = '('   # 원산지 정보는 삭제
    for menu in tmp:
        if search in menu:
            tmp.remove(menu)
    
    lunch.append(tmp)
print("점심 메뉴 행의 수 : ", len(lunch))


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




# print(lunch)

lunch_rice = [] # 점심밥의 메뉴
lunch_main = [] # 점심 메인 메뉴
dinner_rice = [] # 저녁밥의 메뉴
dinner_main = [] # 저녁 메인 메뉴

count = 0
for menu in lunch:
    lunch_rice.append(menu[0])
    lunch_main.append(menu[2])

# for menu in dinner


lunch_rice = pd.DataFrame({ 'lunch_rice': lunch_rice })
lunch_main = pd.DataFrame({ 'lunch_main': lunch_main })


# x_train = train[['요일']]
y1_train = train['중식계']
y2_train = train['석식계']

x_test = test[['요일']]

# print("결측치 확인 1 : \n", x_train.isna().any())
# print("len(x_train) : ", len(x_train))

# 전처리로 생성된 피쳐 : 요일, 연, 월, 일, 실근무인원
# x_train = pd.concat([x_train, train_year, train_month, train_day, train_work_number, lunch_rice, lunch_main], axis=1)

# x_train = [[train['요일'], train_year, train_month, train_day, train_work_number, lunch_rice, lunch_main]]
# x_test = [[test['요일'], test_year, test_month, test_day, test_work_number, lunch_rice, lunch_main]]
x_train = pd.concat([train['요일'], train_year, train_month, train_day, train_work_number, train['본사시간외근무명령서승인건수'], lunch_rice, lunch_main], axis=1, ignore_index = True)
x_test = pd.concat([test['요일'], test_year, test_month, test_day, test_work_number, test['본사시간외근무명령서승인건수'], lunch_rice, lunch_main], axis=1, ignore_index = True)


print("len(train) : ", len(train))
print("len(x_train) : ", len(x_train))
x_train.dropna(axis=0)
print("결측치를 제거한 뒤의 len(x_train) : ", len(x_train))


# x_train = x_train.dropna()
# x_train = x_train.reset_index(drop=True)

print("x_train.loc(931) : ", x_train.loc[931])
print("x_train.loc(932) : ", x_train.loc[932])
print("x_train.loc(934) : ", x_train.loc[934])
print("x_train.loc(933) : ", x_train.loc[933])


'''
i = 0
for count in x_train['근무인원']:
    print(i, " : ", count, " type : ", type(count))
    if count == np.NaN:
        print("nan...")
    i+=1
'''


print("len(train['요일']) = ", len(train['요일']))
print("len(train_year) = ", len(train_year))
print("len(train_month) = ", len(train_month))
print("len(train_day) = ", len(train_day))
print("len(train_work_number) = ", len(train_work_number))
print("len(rice) = ", len(lunch_rice))
print("len(main) = ", len(lunch_main))




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
    'n_estimators' : [200,300,450],
    'max_depth' : [1, 5, 10, 20],
    'max_features' : [ 0.2, 0.5, 0.8, 1]
}

lunch_r = RandomForestRegressor()
dinner_r = RandomForestRegressor()

lunch_model = RandomizedSearchCV(lunch_r, params, scoring='neg_mean_absolute_error')
dinner_model = RandomizedSearchCV(dinner_r, params, scoring='neg_mean_absolute_error')

print("type(x_train) : ", type(x_train))
print("type(y1_train) : ", type(y1_train))

print("len(x_train) : ", len(x_train))
print("len(y1_train) : ", len(y1_train))



'''
print("x_train.loc(0) : ", x_train.loc[[0]])
print("x_train.loc(1000) : ", x_train.loc[[1000]])
print("x_train.loc(1161) : ", x_train.loc[1161])
print("x_train.loc(1162) : ", x_train.loc[1162])
print("x_train.loc(1204) : ", x_train.loc[1204])
'''



lunch_model.fit(x_train, y1_train)
dinner_model.fit(x_train, y2_train)

lunch_best = lunch_model.best_score_
dinner_best = dinner_model.best_score_
print('점심 베이스라인 모델 에러값(mae) : ',lunch_best)
print('저녁 베이스라인 모델 에러값(mae) : ', dinner_best)








'''
model1 = RandomForestRegressor(n_jobs=-1, random_state=42)
model2 = RandomForestRegressor(n_jobs=-1, random_state=42)


model1.fit(x_train, y1_train)
model2.fit(x_train, y2_train)


pred1 = model1.predict(x_test)
pred2 = model2.predict(x_test)


submission['중식계'] = pred1
submission['석식계'] = pred2
'''
print("finish")

# submission.to_csv('baseline.csv', index=False)
