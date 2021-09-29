import pandas as pd

df_pre = pd.read_csv('./dataset/wine.csv', header=None)

df = df_pre.sample(frac=1)

print(df.head(5))