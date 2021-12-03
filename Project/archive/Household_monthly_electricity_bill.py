import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
# import sweetviz as sv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split



'''

num_rooms : Number of room in the house
num_people : Number of people in the house
housearea : Area of the house
is_ac : Is AC present in the house?
is_tv : Is TV present in the house?
is_flat : Is house a flat?
ave_monthly_income : Average monthly income of the household
num_children : Number of children in the house
is_urban : Is the house present in an urban area
amount_paid : Amount paid as the monthly bill

'''

data = pd.read_csv('./Project/archive/Household energy bill data.csv')
data['people'] = data['num_people'] + data['num_children'] # people 수와 아이들의 수를 합하여 하나의 피쳐로 생성함.
data['ac_tv'] = data['is_ac']+data['is_tv'] # 에어컨의 여부와 TV의 여부를 합하여 하나의 피쳐로 생성한다.
data = data.drop(['is_ac','is_tv','num_people','num_children'],axis=1) # 합하여 진 피쳐들을 제거한다.

data = data.drop(['num_rooms', 'housearea', 'is_flat', 'ave_monthly_income'],axis=1)

seed = 0 # seed 값 설정 (동일한 랜덤값을 추출해주기 위해서 시드값을 설정해준다)
np.random.seed(seed)
tf.random.set_seed(3)

# np.round(df, 2)
# print("df : ", type(df))
dataset = data.values



plt.figure(figsize=(12, 12))
sns.heatmap(data.corr(), linewidths=0.1, vmax=0.5, cmap=plt.cm.gist_heat, linecolor="white", annot=True)
plt.show()



''' # 그래프로 분포도 나타냄 
advert_report = sv.analyze(data) 
advert_report.show_notebook() 
'''



X = data.drop(['amount_paid'],axis=1).astype("float")
Y = data['amount_paid']

# X = (dataset[:, 0:9].astype('float32')) / 1 # 9개의 피쳐를 X 변수에 담는다.
# Y = dataset[:, 9] # 한개의 class를 Y 변수에 담는다.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)



model = Sequential()
model.add(Dense(512, input_dim=3, activation="relu")) # 13개의 input과 30개의 output으로 구성된 입력층 생성
model.add(Dense(256, activation="relu")) # 50개의 input과 1개의 output으로 구성된 은닉층 생성
model.add(Dense(256, activation="relu")) 
model.add(Dense(1)) # 선형회귀는 출력층에 활성화함수를 입력하지 않는다.

model.compile(loss="mean_absolute_error", optimizer="adam") 

model.fit(X_train, Y_train, epochs=200, batch_size=10) 

print("X = ", X)
print("Y = ", Y)

print(data.shape)

print("MSE(Training Data) : ", mean_absolute_error(model.predict(X_test), Y_test))



print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)))













'''

# 전체 데이터의 70%를 훈련 데이터로 사용한다.  # 각 샘플을 200번 반복 훈련하며 한번 훈련당 10번씩 훈련한다.

print("X_test : ",  X_test)
print("Y_test : ",  Y_test)

print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))
# print("\n Test Accuracy : ", (model.evaluate(X_test, Y_test)[1]))
'''