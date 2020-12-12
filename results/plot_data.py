import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# ______________________Question 3_____________________________
df = pd.read_csv("results3.csv",usecols=["d", "time"])
print(df)
sns.relplot(x="d", y="time", kind="line", data=df)
plt.show()

# ______________________Question 5_____________________________
df = pd.read_csv("results5.csv",usecols=["d", "time"])
# plt.figure(1)
# plot = 521
N=10
for i in range(6):
    temp_df = df[i*10:(i+1)*10]
    print(temp_df)
    # plt.subplot(plot)
    sns.relplot(x="d", y="time", kind="line", data=temp_df)
    plt.title('N='+str(N))
    # plot+=1
    N*=10

plt.show()
