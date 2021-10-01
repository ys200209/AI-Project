import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# 공부 시간 X와 성적 Y의 리스트 만들기
data = [[2, 0, 81], [4, 4, 93], [6, 2, 91], [8, 3, 97]]
x1 = [i[0] for i in data] 
x2 = [i[1] for i in data]
y = [i[2] for i in data]

# 리스트로 되어 있는 x와 y 값을 넘파이 배열로 바꾸기 (인덱스로 하나씩 불러와 계산할 수 있도록 하기 위함)
x1_data = np.array(x1)
x2_data = np.array(x2)
y_data = np.array(y)

# y = (w1 * x) + (w2 * x) + b
# 기울기 w와 절편 b 초기화 
w1 = 0
w2 = 0
b = 0

# 학습률 
lr = 0.02

# 몇번 반복할 지 설정(0부터 세므로 원하는 반복 횟수에 +1)
epochs = 10

# 경사 하강법 시작
for i in range(epochs): # 2001번 반복 수행
    print("i = ", i)
    print("x1_data = ", x1_data)
    print("x2_data = ", x2_data)
    

    y_pred = (w1 * x1_data) + (w2 * x2_data) + b # y 예측값 구하기
    print("y_pred = ", y_pred)

    error = y_data - y_pred
    print("error = ", error)
    
    # 오차 함수를 w1로 미분한 값
    w1_diff = -(2/len(x1_data)) * sum(x1_data * (error))
    print("w1_diff = ", w1_diff)

    # 오차 함수를 w2로 미분한 값
    w2_diff = -(2/len(x2_data)) * sum(x2_data * (error))
    print("w2_diff = ", w2_diff)

    # 오차 함수를 b로 미분한 값
    b_diff = -(2/len(x1_data)) * sum(y_data - y_pred)
    print("b_diff = ", b_diff)

    w1 = w1 - lr * w1_diff
    w2 = w2 - lr * w2_diff
    b = b - lr * b_diff

    print()
