
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

columns = [f"descriptor_{i}" for i in range(1, 42)] + ["class"]

df = pd.read_csv("biodeg.csv", sep=";", header=None, names=columns)

sns.set_theme(style="whitegrid")

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["class"].value_counts())
print("\nClass percentages:")
print((df["class"].value_counts(normalize=True) * 100).round(2))
print("\nMissing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

duplicates = df[df.duplicated(keep=False)].sort_values(by=columns)
print("\nRows involved in exact duplicates:")
print(duplicates.to_string(index=False))

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="class")
plt.title("Distribution of Biodegradability Classes")
plt.xlabel("Biodegradability Class")
plt.ylabel("Number of Compounds")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.show()

df.iloc[:, :6].hist(figsize=(12, 8), bins=20, edgecolor="black")
plt.suptitle("Distributions of the First Six Molecular Descriptors")
plt.tight_layout()
plt.savefig("descriptor_histograms.png", dpi=150)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df.iloc[:, :10])
plt.title("Boxplots of the First Ten Molecular Descriptors")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("descriptor_boxplots.png", dpi=150)
plt.show()

plt.figure(figsize=(16, 12))
correlation = df.iloc[:, :41].corr()
sns.heatmap(correlation, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Molecular Descriptors")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()

print("\nEDA complete. Graphs saved in your project folder.")


