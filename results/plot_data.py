import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("results.csv",usecols=["d", "time"])
print(df)
df = df[40:50]
print(df)
sns.relplot(x="d", y="time", sort=False, kind="line", data=df)