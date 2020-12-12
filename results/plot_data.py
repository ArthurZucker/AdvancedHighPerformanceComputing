import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("results.csv",usecols=["d", "time"])
N = pd.read_csv("results.csv",usecols=["N"])

# plt.figure(1)
# plot = 521
N=10
for i in range(6):
    temp_df = df[i*10:(i+1)*10]
    # plt.subplot(plot)
    sns.relplot(x="d", y="time", kind="line", data=temp_df)
    plt.title('N='+str(N))
    # plot+=1
    N*=10

plt.show()
