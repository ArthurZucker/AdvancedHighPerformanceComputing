import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
sns.set_style('darkgrid', rc={"lines.linewidth": 2})
sns.set(font_scale=3)

# ______________________Question 1_____________________________

df = pd.read_csv("results1.csv",usecols=["type","memory", "time"])
palette = sns.color_palette("rocket_r")
sns.barplot(
    data=df,
    x="time", y="type",
    hue="memory",palette=palette
)
plt.title('Comparaison des temps de calcul pour mergeSmall_k, avec |M|<1024')
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig("question1.svg", format="svg", bbox_inches='tight')
plt.show()


# ______________________Question 2_____________________________

df = pd.read_csv("results2.csv",usecols=["Kernel","type","memory", "time"])
palette = sns.color_palette("rocket_r")
sns.barplot(
    data=df,
    x="time", y="type",palette=palette
)
plt.title('Comparaison des temps de calcul pour mergeBig_k et pathBig_k')
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig("question2.svg", format="svg", bbox_inches='tight')
plt.show()



# ______________________Question 3_____________________________
df = pd.read_csv("results3.csv",usecols=["d", "time"])
print(df)
sns.lineplot(x="d", y="time", data=df)

plt.title('Comparaison des temps de calcul pour merge sort en fonction de d')
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig("question3.svg", format="svg", bbox_inches='tight')
plt.show()


# ______________________Question 4_____________________________

df = pd.read_csv("results4.csv",usecols=["type","memory", "time"])
palette = sns.color_palette("rocket_r")
sns.barplot(
    data=df,
    x="time", y="type",
    hue="memory",palette=palette
)
plt.title('Comparaison des temps de calcul pour mergeBatch_k en fonction des type de memoire')
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig("question4.svg", format="svg", bbox_inches='tight')
plt.show()


# ______________________Question 5_____________________________
df = pd.read_csv("results5.csv",usecols=["N","d", "time"])
# plt.figure(1)
# plot = 521
N=10

palette = sns.color_palette("rocket_r")

# Plot the lines on two facets
sns.lineplot(
    data=df,
    x="d", y="time",
    hue="N"
)
plt.title('Comparaison des temps de calcul pour mergeBatch_k en fonction de d')
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig("question4.svg", format="svg", bbox_inches='tight')
plt.show()


