# 최소 제곱법을 이용한 기울기 a와 y의 절편을 구하는 방법.
import numpy as np

# x 값과 y 값
x = [2, 4, 6, 8]
y = [81, 93, 91, 97]

# x와 y의 평균값
mx = np.mean(x)
my = np.mean(y)
print("x = ", x)
print("y = ", y)
print("x의 평균값 : " , mx)
print("y의 평균값 : " , my)

'''
       (x-x평균)(y-y평균)의 합
a  =  ------------------------
       ((x - x평균)의 제곱)의 합
'''

# 기울기 공식의 분모 (기울기 a)
divisor1 = sum([(i-mx) ** 2 for i in x]) # x - x평균의 제곱
divisor2 = sum([(mx-i) ** 2 for i in x]) # x평균 - x의 제곱 ------

# 기울기 공식의 분자 (기울기 a)
def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d += (x[i]-mx) * (y[i]-my)
    return d

dividend = top(x, mx, y, my)

print("분모1 : ", divisor1)
# print("분모2 : ", divisor2)
print("분자 : ", dividend)

'''
y 절편 = y의 평균 - (x의 평균 * 기울기 a)
'''
# 기울기와 y 절편 구하기
a = dividend / divisor1
b = my - (mx * a) 

print("기울기 a = ", a)
print("y 절편 b = ", b)
