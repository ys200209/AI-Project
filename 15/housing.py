import pandas as pd

df = pd.read_csv('./dataset/housing.csv', delim_whitespace=True,
    header=None)

print(df.info())

print("---------------------------------------------------------")

print(df.head())

model = Sequential()

Y_prediction = model.predict(X_test).flatten() # 데이터 배열이 몇 차원이든 모두 1차원으로 바꿔주는 함수

