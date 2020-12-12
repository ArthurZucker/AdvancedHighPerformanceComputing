import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("results3.csv",usecols=["d", "time"])

# plt.figure(1)
# plot = 521
temp_df = df[i*10:(i+1)*10]
# plt.subplot(plot)
print(temp_df)
sns.relplot(x="d", y="time", kind="line", data=temp_df)
# plot+=1
plt.show()

# df = pd.read_csv("results5.csv",usecols=["d", "time"])
# # plt.figure(1)
# # plot = 521
# N=10
# for i in range(6):
#     temp_df = df[i*10:(i+1)*10]
#     # plt.subplot(plot)
#     sns.relplot(x="d", y="time", kind="line", data=temp_df)
#     plt.title('N='+str(N))
#     # plot+=1
#     N*=10

# plt.show()


