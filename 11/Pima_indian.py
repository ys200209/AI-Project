import pandas as pd

df = pd.read_csv('./dataset/pima-indians-diabetes.csv',
                names=["pregnant", "plasma", "pressure", "thickness", 
                "insulin", "BMI", "predigree", "age", "class"])
# csv 파일에는 데이터를 설명하는 한 줄의 헤더(header)가 존재하지만 위의 파일에는 헤더가 존재하지 않는다.
# 따라서 이에 names라는 함수를 통해 속성별 키워드(열 이름)를 지정해 주었다.

print(df.head(5)) # 해당 csv 파일의 첫 다섯 줄을 불러온다.

print("----------------------------------------------------------------")

print(df.info()) # 데이터의 전반적인 정보를 확인하는 .info() 함수

print("----------------------------------------------------------------")

print(df[['pregnant', 'class']].groupby(['pregnant'], as_index=False).mean().sort_values(by="pregnant", ascending=True))
# 임신 횟수와 당뇨병 발병 확률의 상관관계
# 1. groupby() 함수를 사용해 임신 횟수 정보를 기준으로 하는 그룹을 생성.
# 2. as_index=False 를 통해 pregnant 정보 옆에 새로운 인덱스를 생성.
# 3. mean() 함수를 사용해 평균을 구하고 임신 횟수에 대한 오름차순 정렬로 정보를 가공한다.


