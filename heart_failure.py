
import os
os.chdir("/Users/clothildedugros/Documents/Python")

import pandas as pd
data = pd.read_csv("./Heart Failure/heart_failure.csv", sep=",")
import numpy as np
# visualisation
import seaborn as sns
import matplotlib.pyplot as plt

## Visualisation of the data
print(data.head(5))
print(data.shape) # 299 observations and 13 variables
print(data.dtypes)
# age platelets, serum_creatine are float 64, the rest are int64
#transform to int8 ? 


## Clean Data 
print(data.info())
desc = data.describe()

# No NAs
#duplicates ? 
print(data.duplicated()) # No dupliates

# Data manipulation 
data["age"] = data["age"].astype("int8")

## Mortality Rate

# Let's study the variable "DEATH_EVENT"
print(data["DEATH_EVENT"].value_counts())

# Compute the  mortality rate due to heart failure
mortality_rate = (data.DEATH_EVENT.sum()/len(data.DEATH_EVENT))*100
print("The mortality rate due to heart failure among the study's participants is {rate} percent.".format(rate=round(mortality_rate,2)))

# Sub-groups analysis 

# sex
100*pd.crosstab(data["DEATH_EVENT"], data["sex"], normalize="index")

# age
death_rate_age= 100*pd.crosstab(data["DEATH_EVENT"], data["age"], normalize="index")
#Let's do break down age into classes 

bins = [0, 30, 50, 70, 100]
labels = ['Young', 'Middle-aged', 'Senior', 'Elderly']

# Use pd.cut() to assign age categories
age_classes = pd.cut(age, bins=bins, labels=labels, right=False)
print(data["age"].value_counts())



# ANAEMIA 
#Whether the patient has anemia (decrease of red blood cells or hemoglobin) or not. 

# Count
fig, ax = plt.subplots(figsize=(8, 5))

sns.countplot(
    data=data,
    x="anaemia"
  )
ax.set(
  title="Whether the patient has anaemia",
  xlabel="anaemia",
  ylabel="Count"
)

sns.despine()

# Freq 
anaemia_counts = data["anaemia"].value_counts()
# more than half ot the sample has no anemia (deacrese of red blood cells or hemoglobin) -> 57% 

plt.bar('Total People', anaemia_counts[0], label='No Anaemia', color='blue')
plt.bar('Total People', anaemia_counts[1], bottom=anaemia_counts[0], label='Anaemia', color='red')
plt.ylabel('Count of People')
plt.title('Count of People with and without Anaemia')
plt.legend(title='Anaemia Status')
plt.show()

# Pie chart 
anameia_labels = ['No Anaemia', 'Anaemia']
plt.figure(figsize=(8,8))
plt.pie(
    x= anaemia_counts,
    labels=anameia_labels,
    autopct='%1.2f%%',
    textprops={'fontsize':14},
    colors=[
        '#85C1E9', '#EC7063'
        ])
hole = plt.Circle((0, 0), 0.4, facecolor='white') # donut chart
plt.gcf().gca().add_artist(hole)
plt.title("Whether the patients have anaemia")
plt.show()


fig, ax = plt.subplots()
ax.pie(anaemia_counts, labels=anameia_labels)
ax.set(
  title="Whether the patients have anaemia",
)

# DIABETES
diabetes_counts = data["diabetes"].value_counts()
# Diabetes - 0 = No, 1 = Yes
diabetes_labels = ['No diabetes', 'Diabetes']

plt.figure(figsize=(8,8))
plt.pie(
    x= diabetes_counts,
    labels=diabetes_labels,
    autopct='%1.2f%%',
    textprops={'fontsize':14},
    colors=[
        '#85C1E9', '#EC7063'
        ])
hole = plt.Circle((0, 0), 0.4, facecolor='white') # donut chart
plt.gcf().gca().add_artist(hole)
plt.title("Whether the patients have diabetes")
plt.show()

## High blood pressure
#0 = No, 1 = Yes
blood_pressure_counts = data["high_blood_pressure"].value_counts()
blood_pressure_labels = ['No high blodd pressure', 'High blood pressure']

plt.figure(figsize=(8,8))
plt.pie(
    x= blood_pressure_counts,
    labels=blood_pressure_labels,
    autopct='%1.2f%%',
    textprops={'fontsize':14},
    colors=[
        '#85C1E9', '#EC7063'
        ])
hole = plt.Circle((0, 0), 0.4, facecolor='white') # donut chart
plt.gcf().gca().add_artist(hole)
plt.title("Whether the patients have diabetes")
plt.show()

# serum_creatinine

fig, ax = plt.subplots(figsize=[7, 5])
sns.set_theme(style="white")

sns.kdeplot(
    data=data,
    x="serum_creatinine",
    color="steelblue"
)

ax.set(
    title="serum creatinine distribution",
    xlabel="Level of serum creatinine in the blood (mg/dL). "
)

sns.despine(offset=5)

# SEX 
#which one is 1 and 0 ? 
# assume 1 is male and 0 is female 
data["sex"].value_counts(normalize=True)
sex_age = data.groupby("sex").agg(median_age=("age", "median"))

sex_counts = data["sex"].value_counts()
sex_labels = ['Men', 'Women']
plt.figure(figsize=(8,8))
plt.pie(
    x= sex_counts,
    labels=sex_labels,
    autopct='%1.2f%%',
    textprops={'fontsize':14},
    colors=[
        '#85C1E9', '#EC7063'
        ])
hole = plt.Circle((0, 0), 0.4, facecolor='white') # donut chart
plt.gcf().gca().add_artist(hole)
plt.title("Sex of the patients")
plt.show()

# time 



## HEATMAP
sns.set_theme(style="white")

# Compute the correlation matrix
corr = data.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# GROUPED BAR PLOTS
sns.set_theme(style="whitegrid")

# Draw a nested barplot by species and sex
g = sns.catplot(
    data=data, kind="bar",
    x="diabetes", y="anaemia", hue="sex")
g.despine(left=True)
g.set_axis_labels("", "Body mass (g)")
g.legend.set_title("")

