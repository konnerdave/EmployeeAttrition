import numpy as np  # Linear algebra
import pandas as pd  # Data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For sub plotting graphs

"""
Created on Sun Aug 30 01:18:39 2020

@author: Himalaya Dave
"""

# Data attributes of our dataset
attributes = [
    "Age",
    "Attrition",
    "Business Travel",
    "DailyRate",
    "Department",
    "Distance From Home",
    "Education",
    "Education Field",
    "Employee Count",
    "Employee Number",
    "EnvironmentSatisfaction",
    "Gender",
    "Hourly Rate",
    "Job Involvement",
    "Job Level",
    "Job Role",
    "Job Satisfaction",
    "Marital Status",
    "Monthly Income",
    "Num Companies Worked",
    "Over18",
    "OverTime",
    "Percent Salary Hike",
    "Performance Rating",
    "Relationship Satisfaction",
    "Standard Hours",
    "Stock Option Level",
    "Total Working Years",
    "Training Times LastYear",
    "Work Life Balance",
    "Years At Company",
    "Years In Current Role",
    "Years Since Last Promotion",
    "Years With Curr Manager",
]

# Loading dataset
pd.set_option("display.max_rows", None, "display.max_columns", 34)
dataset = pd.read_csv("../Input/AttritionData.csv")


# Checking attributes names and datatypes
print("****************Dataset Info****************")
dataset.info()
print("****************END****************")


# Data insights
left = dataset.groupby("Attrition")
left.mean()
print("****************Left Mean****************")
print(left.mean())
print("****************END****************")


# Summary statistics
dataset.describe()
print("****************Summary statistics****************")
print(dataset.describe())
print("****************END****************")


# Employees Left Graph
# The bar graph is suitable for showing discrete variable counts.
left_count = left.count()
plt.bar(left_count.index.values, left_count["Job Satisfaction"])
plt.xlabel("Employees Left Company")
plt.ylabel("Number of Employees")
plt.show()

# No. of employees left
dataset.Attrition.value_counts()
print("****************No. of employees left****************")
print(dataset.Attrition.value_counts())
print("****************END****************")


# Time Spent in Company Graph
time_spent = dataset.groupby("Years At Company").count()
plt.bar(time_spent.index.values, time_spent["Job Satisfaction"])
plt.xlabel("Number of Years Spend in Company")
plt.ylabel("Number of Employees")
plt.show()


# Subplots using Seaborn
features = [
    "Attrition",
    "Business Travel",
    "Job Satisfaction",
    "OverTime",
    "Percent Salary Hike",
    "Performance Rating",
    "Years In Current Role",
    "Years Since Last Promotion",
]

# Graph subplots of Features & Emloyees
fig = plt.subplots(figsize=(35, 40))

for i, j in enumerate(features):
    plt.subplot(4, 2, i + 1)
    plt.subplots_adjust(hspace=1.0)
    sns.countplot(x=j, data=dataset)
    plt.xticks(rotation=90)
    plt.title("No. of Employees")
plt.show()


# Graph subplots of Feature-Attrition & Employees
fig = plt.subplots(figsize=(35, 40))

for i, j in enumerate(features):
    plt.subplot(4, 2, i + 1)
    plt.subplots_adjust(hspace=1.0)
    sns.countplot(x=j, data=dataset, hue="Attrition")
    plt.xticks(rotation=90)
    plt.title("No. of Employees")
plt.show()


from sklearn.cluster import KMeans
# Filter dataset
left_emp =  dataset[["Job Satisfaction", "Years Since Last Promotion"]][dataset.Attrition == "Yes"]
# Create groups using K-means clustering.
kmeans = KMeans(n_clusters = 3, random_state = 0).fit(left_emp)

# Add new column "label" annd assign cluster labels.
left_emp["label"] = kmeans.labels_
# Draw scatter plot
plt.scatter(left_emp["Job Satisfaction"], left_emp["Years Since Last Promotion"], c=left_emp["label"],cmap="Accent")
plt.xlabel("Satisfaction Level")
plt.ylabel("Last Evaluation")
plt.title("3 Clusters of employees who left")
plt.show()
