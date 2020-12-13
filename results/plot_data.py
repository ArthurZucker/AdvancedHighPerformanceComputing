import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
sns.set_style('darkgrid', rc={"lines.linewidth": 2})
sns.set(font_scale=3)
plot_backend = matplotlib.get_backend()
mng = plt.get_current_fig_manager()
if plot_backend == 'TkAgg':
    mng.resize(*mng.window.maxsize())
elif plot_backend == 'wxAgg':
    mng.frame.Maximize(True)
elif plot_backend == 'Qt4Agg':
    mng.window.showMaximized()
# # ______________________Question 1_____________________________

# df = pd.read_csv("results1.csv",usecols=["type","memory", "time"])
# palette = sns.color_palette("rocket_r")
# sns.barplot(
#     data=df,
#     x="time", y="type",
#     hue="memory",palette=palette
# )
# plt.title('Comparison of the execution time for mergeSmall_k, with |M|<1024')
# mng = plt.get_current_fig_manager()
# mng.full_screen_toggle()
# plt.savefig("question1.svg", format="svg", bbox_inches='tight')
# plt.show()


# ______________________Question 2_____________________________

df = pd.read_csv("results2.csv",usecols=["Kernel","type","memory", "time"])
palette = sns.color_palette("rocket_r")

g=sns.catplot(
    data=df,
    x="Kernel", y="time",hue="memory",col="type",palette=palette,kind="bar",ci=None,sharey=False
)
(g.set_axis_labels("", "time")
.set_titles("{col_name} {col_var}")
.despine(left=True)
) 
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig("question2.svg", format="svg", bbox_inches='tight')
plt.show()


df = pd.read_csv("results4.csv",usecols=["col","type","memory", "time"])
palette = sns.color_palette("rocket_r")
palette = sns.color_palette("rocket_r")

g=sns.catplot(
    data=df,
    x="type", y="time",hue="memory",col="col",palette=palette,kind="bar",ci=None,sharey=False,sharex=False
)
(g.set_axis_labels("", "time")
.set_titles("{col_name} {col_var}")
.despine(left=True)
) 
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig("question4b2.svg", format="svg", bbox_inches='tight')
plt.show()

# # ______________________Question 3_____________________________
# df = pd.read_csv("results3.csv",usecols=["d", "time"])
# print(df)
# sns.lineplot(x="d", y="time", data=df)

# plt.title('Comparison of the execution time for merge sort w.r.t d')
# mng = plt.get_current_fig_manager()
# mng.full_screen_toggle()
# plt.savefig("question3.svg", format="svg", bbox_inches='tight')
# plt.show()


# ______________________Question 4_____________________________

df = pd.read_csv("results4.csv",usecols=["col","type","memory", "time"])
palette = sns.color_palette("rocket_r")

fig_dims = (25, 14)
fig, axes = plt.subplots(1, 2,figsize=fig_dims)
fig.suptitle('Comparison of the execution time for mergeSmallBatch_k w.r.t the type of memory')

sns.catplot(ax=axes[0],
    data=df.loc[df['col'] == "one"],
    x="type", y="time",
    hue="memory",col="col",palette=palette,kind="bar"
)

sns.catplot(ax=axes[1],
    data=df.loc[df['col'] == "two"],
    x="type", y="time",
    hue="memory",col="col",palette=palette,kind="bar"
)
fig.savefig("question4.svg", format="svg", bbox_inches='tight')
plt.close()





sns.set(rc={'figure.figsize':(25,14)})
df = pd.read_csv("results4.csv",usecols=["col","type","memory", "time"])
palette = sns.color_palette("rocket_r")
fig_dims = (25, 14)
plt.figure(figsize=fig_dims)
sns.catplot(
    data=df.loc[df['memory'] == "ZeroCpy"],
    x="time", y="type",palette=palette,kind="bar"
)
plt.title('Comparison of the execution time for mergeSmallBatch_k w.r.t the type of memory')
plt.savefig("question4b.svg", format="svg", bbox_inches='tight')
plt.close()


# # ______________________Question 5_____________________________
# df = pd.read_csv("results5.csv",usecols=["N","d","time"])
# df["N"] = df["N"].astype(str)
# sns.lineplot(
#     data=df,
#     x="d", y="time",
#     hue="N",palette=["b", "g", "r", "indigo", "k","cyan"]
# )
# plt.title('Comparison of the execution time for mergeSmallBatch_k (using shared memory) w.r.t d')
# mng = plt.get_current_fig_manager()
# mng.full_screen_toggle()
# plt.savefig("question5.svg", format="svg", bbox_inches='tight')
# plt.show()
