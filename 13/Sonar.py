import pandas as pd

df = pd.read_csv('./dataset/sonar.csv', header=None)
print(df.info())

print(df.head())