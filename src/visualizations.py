import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlations(df):
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

def plot_feature_vs_success(df, feature):
    plt.figure(figsize=(6,4))
    sns.boxplot(x="success", y=feature, data=df)
    plt.title(f"{feature} vs Success")
    plt.show()
