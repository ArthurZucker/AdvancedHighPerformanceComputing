import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
sns.set_style('darkgrid')

# ______________________Question 1_____________________________

df = pd.read_csv("results1.csv",usecols=["type","memory", "time"])
palette = sns.color_palette("rocket_r")
sns.barplot(
    data=df,
    x="time", y="type",
    hue="memory",palette=palette,
    height=5, aspect=.75
)
plt.show()


# ______________________Question 2_____________________________

df = pd.read_csv("results2.csv",usecols=["Kernel","type","memory", "time"])
palette = sns.color_palette("rocket_r")
sns.barplot(
    data=df,
    x="time", y="type",palette=palette,
    height=5, aspect=.75, facet_kws=dict(sharex=False),
)
plt.show()



# ______________________Question 3_____________________________
df = pd.read_csv("results3.csv",usecols=["d", "time"])
print(df)
sns.relplot(x="d", y="time", kind="line", data=df)
plt.show()
plt.savefig("question3.svg", format="svg")

# ______________________Question 1_____________________________

df = pd.read_csv("results4.csv",usecols=["type","memory", "time"])
palette = sns.color_palette("rocket_r")
sns.barplot(
    data=df,
    x="time", y="type",
    hue="memory",palette=palette,
    height=5, aspect=.75
)
plt.show()


# ______________________Question 5_____________________________
df = pd.read_csv("results5.csv",usecols=["N","d", "time"])
# plt.figure(1)
# plot = 521
N=10

palette = sns.color_palette("rocket_r")

# Plot the lines on two facets
sns.relplot(
    data=df,
    x="d", y="time",
    hue="N",
    kind="line", palette=palette,
    height=5, aspect=.75, facet_kws=dict(sharex=False),
)
plt.show()

for i in range(6):
    temp_df = df[i*10:(i+1)*10]
    print(temp_df)
    # plt.subplot(plot)
    sns.relplot(x="d", y="time", kind="line", data=temp_df)
    plt.title('N='+str(N))
    # plot+=1
    N*=10

plt.show()

