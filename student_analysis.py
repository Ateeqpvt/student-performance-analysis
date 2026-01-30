# ================================
# Student Performance Analysis
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load Dataset
# -------------------------------
data = pd.read_csv("data/students.csv")

print("First 5 rows of dataset:")
print(data.head())

# -------------------------------
# 2. Basic Information
# -------------------------------
print("\nDataset Info:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())

# -------------------------------
# 3. Check Missing Values
# -------------------------------
print("\nMissing Values:")
print(data.isnull().sum())

# -------------------------------
# 4. Feature Engineering
# Create Total Score
# -------------------------------
data["total_score"] = (
    data["math score"] +
    data["reading score"] +
    data["writing score"]
)

print("\nTotal Score Column Added:")
print(data[["total_score"]].head())

# -------------------------------
# 5. Analysis
# -------------------------------

# Average subject scores by gender
avg_gender = data.groupby("gender")[["math score", "reading score", "writing score"]].mean()
print("\nAverage Scores by Gender:")
print(avg_gender)

# Effect of test preparation
prep_effect = data.groupby("test preparation course")[["math score", "reading score", "writing score"]].mean()
print("\nEffect of Test Preparation:")
print(prep_effect)

# Average total score by gender
total_by_gender = data.groupby("gender")["total_score"].mean()
print("\nAverage Total Score by Gender:")
print(total_by_gender)

# Identify top performers
top_students = data[data["total_score"] > 250]
print("\nNumber of High Performing Students (Total Score > 250):")
print(top_students.shape[0])

# -------------------------------
# 6. Correlation Analysis
# -------------------------------
correlation = data[["math score", "reading score", "writing score"]].corr()
print("\nCorrelation Matrix:")
print(correlation)

# -------------------------------
# 7. Visualizations
# -------------------------------

# Bar chart: Average Math Score by Gender
sns.barplot(x="gender", y="math score", data=data)
plt.title("Average Math Score by Gender")
plt.show()

# Bar chart: Total Score by Test Preparation
sns.barplot(x="test preparation course", y="total_score", data=data)
plt.title("Total Score by Test Preparation")
plt.show()

# Heatmap: Subject Correlation
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Between Subject Scores")
plt.show()

# -------------------------------
# 8. Save Cleaned Data
# -------------------------------
data.to_csv("data/cleaned_students.csv", index=False)
print("\nCleaned dataset saved as cleaned_students.csv")

# -------------------------------
# END OF PROJECT
# -------------------------------
