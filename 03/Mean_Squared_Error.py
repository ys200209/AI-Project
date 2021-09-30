import numpy as np

# 기울기 a와 y 절편 b
fake_a_b = [3, 76]

# x, y의 데이터 값
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

# y = ax + b에 a와 b 값을 대입하여 결과를 출력하는 함수 ( a와 b값을 대입한 '예측값' )
def predict(x):
    return fake_a_b[0]*x + fake_a_b[1] # 3과 76을 대입

# MSE(Mean_Squared_Error: 평균 제곱 오차)
def mse(y, y_hat): # 실제 y값과 예측 y값을 매개변수로 받아옴
    return ((y-y_hat)**2).mean()

# MSE 함수를 각 y 값에 대입하여 최종 값을 구하는 함수
def mse_val(y, predict_result): # 
    return mse(np.array(y), np.array(predict_result))

# 예측 값이 들어갈 빈 리스트
predict_result = []

# 모든 x 값을 한 번씩 대입하여
for i in range(len(x)):
    # predict_result 리스트를 작성
    predict_result.append(predict(x[i]))
    print("공부한 시간=%.f, 실제 점수=%.f, 예측 점수=%.f" % (x[i], y[i], predict(x[i])))

# 최종 MSE 출력
print("MSE 최종값: " + str(mse_val(y, predict_result)))




