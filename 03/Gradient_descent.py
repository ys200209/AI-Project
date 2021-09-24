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

# 리스트로 되어 있는 x와 y값을 넘파이 배열로 바꾸기 ( 인덱스를 주어 하나씩 불러와 계산이 가능하게 하기 위함 )
x_data = np.array(x)
y_data = np.array(y)

# 기울기 a와 절편 b의 값 초기화
a = 0
b = 0

# 학습률 정하기
lr = 0.03

# 몇 번 반복 학습할건지 설정
epochs = 2001 # 2001번 반복 학습

# 경사 하강법 시작
for i in range(epochs): # 에포크 수만큼 반복 
    y_pred = a * x_data + b # y를 구하는 식 세우기 (y = ax + b)
    error = y_data - y_pred # 예측값 - 실제값을 통해 오차를 구하는 식 ( y_data: 실제 점수값, y_pred: 위의 식을 통해 도출된 예측값)
    # 오차 함수를 a로 미분한 값
    a_diff = -(2/len(x_data)) * sum(x_data * (error))
    # 오차 함수를 b로 미분한 값
    b_diff = -(2/len(x_data)) * sum(error)

    a = a - lr * a_diff # 학습률을 곱해 기존의 a값 업데이트
    b = b - lr * b_diff # 학습률을 곱해 기존의 b값 업데이트

    if (i % 100) == 0: # 100번 학습할 때마다 현재의 예측값 출력
        print("[학습량]=%.f, [기울기]=%.04f, [y절편]=%.04f" % (i, a, b))

# 최종적으로 학습한 기울기와 절편을 이용한 예측 그래프 그리기
y_pred = a * x_data + b
plt.scatter(x, y)
plt.plot([min(x_data), max(x_data)], [min(y_pred), max(y_pred)])
plt.show() 









