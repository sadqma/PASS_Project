import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_correlations(df):
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

def plot_feature_vs_success(df, feature):
    df[feature] = pd.to_numeric(df[feature], errors='coerce')
    if df["success"].dtype != "object":
        df["success"] = df["success"].replace({1: "Passed", 0: "Failed"})

    df = df.dropna(subset=[feature, "success"])

    plt.figure(figsize=(10, 6))
    success_groups = df["success"].unique()
    colors = sns.color_palette("Dark2", len(success_groups))

    for i, s in enumerate(success_groups):
        group_data = df[df["success"] == s][feature]
        plt.hist(group_data, bins=15, density=True, alpha=0.8, color=colors[i], label=f"{s} histogram")
        sns.rugplot(group_data, color=colors[i], alpha=0.8)

        sns.kdeplot(group_data, color=colors[i], alpha=0.5, linewidth=2, label=f"{s} density")

        mean_val = group_data.mean()
        median_val = group_data.median()
        plt.axvline(mean_val, color=colors[i], linestyle='-', linewidth=2)
        plt.axvline(median_val, color=colors[i], linestyle='--', linewidth=2)

        plt.text(mean_val, plt.ylim()[1] * 0.95 - i * 0.05 * plt.ylim()[1], f"Mean {s}: {mean_val:.1f}",
                 color="black")
        plt.text(median_val, plt.ylim()[1] * 0.87 - i * 0.05 * plt.ylim()[1], f"Median {s}: {median_val:.1f}",
                 color="black")
        print(f"{s}: mean = {mean_val:.2f}, median = {median_val:.2f}, n={len(group_data)}")

    plt.title(f"Distribution of {feature} by success", fontsize=16)
    plt.xlabel(f"{feature}%", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.grid(alpha=0.3)

    plt.ylim(bottom=0)
    plt.xlim(60, 100)
    plt.legend(loc="upper right", fontsize=12)
    plt.show()
