# 경사 하강법: 2차함수 f(x)에서 기울기 a를 변화시키며 최솟값 m을 찾아내는 방법

# 최솟값을 구하기 위해서는 이차 함수에서 미분을 해야하는데
# 그 이차 함수는 평균 제곱 오차(MSE: Mean Square Error)를 통해 나온다.
# 때문에 평균 제곱 오차를 a와 b로 각각 편미분 해야 한다.

# 평균 제곱 오차(MSE) : ((예측값 - 실제값)의 제곱)의 합을 n으로 나눈 평균 값

# 곱 미분의 공식에 의한 미분 값
# a로 편미분 한 결과 : 2/n 시그마(ax + b - y)*x
# b로 편미분 한 결과 = 2/n 시그마(ax + b - y)

'''
y_pred = a * x_data + b ( 오차 함수인 y = ax + b를 정의한 부분 )
error = y_data - y_pred ( 실제값 - 예측값, 즉 오차를 구하는 식 )

# 평균 제곱 오차를 a로 미분한 결과
a_diff = -(2 / len(x_data)) * sum(x_data * error))

# 평균 제곱 오차를 b로 미분한 결과
b_diff = -(2 / len(x_data)) * sum(error)

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 공부 시간 X와 성적 Y의 리스트를 만들기
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

# 그래프로 나타내기
plt.figure(figsize=(8, 5))
plt.scatter(x, y)
plt.show()











