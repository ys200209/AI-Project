import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# 공부 시간 X와 성적 Y의 리스트 만들기
data = [[2, 0, 81], [4, 4, 93], [6, 2, 91], [8, 3, 97]]
x1 = [i[0] for i in data]
x2 = [i[1] for i in data]
y = [i[2] for i in data]

# 그래프로 확인
ax = plt.axes(projection='3d')
ax.set_xlabel('study_hours')
ax.set_ylabel('private_class')
ax.set_zlabel('Score')
ax.dist = 11
ax.scatter(x1, x2, y)
plt.show()

# 리스트로 되어 있는 x와 y 값을 넘파이 배열로 바꾸기 (인덱스로 하나씩 불러와 계산할 수 있도록 하기 위함)
x1_data = np.array(x1)
x2_data = np.array(x2)
y_data = np.array(y)

# 기울기 a와 절편 b의 값 초기화
a1 = 0 # x1의 기울기
a2 = 0 # x2의 기울기
b = 0

# 학습률
lr = 0.02

# 몇 번 반복 학습할지 설정
epochs = 2001

# 경사 하강법 시작
for i in range(epochs):
    y_pred = a1 * x1_data + a2 * x2_data + b # y를 구하는 식 세우기
    error = y_data - y_pred # 오차를 구하는 식 (실제값 - 예측값)
    # 오차 함수를 a1로 미분한 값
    a1_diff = -(2/len(x1_data)) * sum(x1_data * (error))
    # 오차 함수를 a2로 미분한 값
    a2_diff = -(2/len(x2_data)) * sum(x2_data * error)
    # 오차 함수를 b로 미분한 값
    b_diff = -(2/len(x1_data)) * sum(error)

    a1 = a1 - lr * a1_diff # 학습률을 곱해 기존의 값을 업데이트
    a2 = a2 - lr * a2_diff
    b = b - lr * b_diff

    if (i % 100) == 0:
        print("[학습량]=%.f, [기울기1]=%.04f, [기울기2]=%.04f [y절편]=%.04f" % (i, a1, a2, b)) 

'''
ax.scatter(x1, x2, y)
#plt.plot([min(x1_data), max(x1_data)], [min(x2_data), max(x2_data)], [min(y_pred), max(y_pred)])
plt.show() 
'''