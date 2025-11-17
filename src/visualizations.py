import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.inspection import DecisionBoundaryDisplay


def plot_correlations(df):
    # Берём только числовые колонки
    numeric_df = df.select_dtypes(include=["number"])

    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap (Numeric Features Only)")
    plt.tight_layout()
    plt.show()


def plot_feature_vs_success(df, feature):
    # Преобразуем колонку в числовой формат
    df[feature] = pd.to_numeric(df[feature], errors='coerce')

    # Если success — числовой, переведём в текст
    if df["success"].dtype != "object":
        df["success"] = df["success"].replace({1: "Passed", 0: "Failed"})

    # Убираем строки с пропущенными значениями
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

        plt.text(mean_val, plt.ylim()[1] * 0.95 - i * 0.05 * plt.ylim()[1],
                 f"Mean {s}: {mean_val:.1f}", color="black")
        plt.text(median_val, plt.ylim()[1] * 0.87 - i * 0.05 * plt.ylim()[1],
                 f"Median {s}: {median_val:.1f}", color="black")

        print(f"{s}: mean = {mean_val:.2f}, median = {median_val:.2f}, n={len(group_data)}")

    plt.title(f"Distribution of {feature} by success", fontsize=16)
    plt.xlabel(f"{feature}%", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.grid(alpha=0.3)
    plt.ylim(bottom=0)
    plt.xlim(60, 100)
    plt.legend(loc="upper right", fontsize=12)
    plt.show()





def other_plot(df):
    df_plot = df.dropna(subset=["attendance", "assignments_completed",
                                "participation", "LMS_activity", "final_grade", "success"])
    # Ensure success is categorical
    df_plot["success"] = df_plot["success"].replace({1: "Passed", 0: "Failed"})

    features = ['attendance', 'assignments_completed', 'participation', 'LMS_activity']
    X = df_plot[features]
    y = df_plot['success'].map({'Passed': 1, 'Failed': 0})


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3,
                                                        random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(random_state=42)

    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    # Predictions
    y_pred_rf = rf.predict(X_test)
    y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]
    y_pred_lr = lr.predict(X_test)
    y_pred_proba_lr = lr.predict_proba(X_test)[:, 1]


    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Confusion Matrix - Random Forest
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Random Forest - Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')


    cm_lr = confusion_matrix(y_test, y_pred_lr)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', ax=axes[0, 1])
    axes[0, 1].set_title('Logistic Regression - Confusion Matrix')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')

    # ROC Curve
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    roc_auc_lr = auc(fpr_lr, tpr_lr)

    axes[1, 0].plot(fpr_rf, tpr_rf, color='darkorange', lw=2,
                    label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
    axes[1, 0].plot(fpr_lr, tpr_lr, color='green', lw=2,
                    label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
    axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1, 0].set_xlim([0.0, 1.0])
    axes[1, 0].set_ylim([0.0, 1.05])
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curve - Model Comparison')
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].grid(True)
    # Feature Importance
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=True)

    axes[1, 1].barh(importance_df['feature'], importance_df['importance'], color='skyblue')
    axes[1, 1].set_xlabel('Feature Importance')
    axes[1, 1].set_title('Random Forest Feature Importance')
    for i, v in enumerate(importance_df['importance']):
        axes[1, 1].text(v + 0.01, i, f'{v:.3f}', va='center')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))

    # Create dataframe with probabilities
    prob_df = pd.DataFrame({
        'True_Label': y_test.map({1: 'Passed', 0: 'Failed'}),
        'RF_Probability': y_pred_proba_rf,
        'LR_Probability': y_pred_proba_lr
    })

    plt.subplot(1, 2, 1)
    for label in ['Passed', 'Failed']:
        data = prob_df[prob_df['True_Label'] == label]['RF_Probability']
        plt.hist(data, bins=15, alpha=0.7, label=label, density=True)
    plt.xlabel('Prediction Probability (Random Forest)')
    plt.ylabel('Density')
    plt.title('RF: Prediction Probability by True Label')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    for label in ['Passed', 'Failed']:
        data = prob_df[prob_df['True_Label'] == label]['LR_Probability']
        plt.hist(data, bins=15, alpha=0.7, label=label, density=True)
    plt.xlabel('Prediction Probability (Logistic Regression)')
    plt.ylabel('Density')
    plt.title('LR: Prediction Probability by True Label')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    top_features_idx = np.argsort(rf.feature_importances_)[-2:]
    top_features = [features[i] for i in top_features_idx]

    X_top = X_scaled[:, top_features_idx]



    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    rf_2d = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_2d.fit(X_top, y)

    DecisionBoundaryDisplay.from_estimator(
        rf_2d, X_top, response_method="predict",
        alpha=0.5, ax=axes[0],
        xlabel=f"{top_features[0]} (scaled)",
        ylabel=f"{top_features[1]} (scaled)",
    )

    scatter = axes[0].scatter(X_top[:, 0], X_top[:, 1], c=y, edgecolors='black',
                              s=50, cmap='coolwarm')
    axes[0].set_title(f'Random Forest Decision Boundary\n{top_features[0]} vs {top_features[1]}')
    axes[0].legend(handles=scatter.legend_elements()[0], labels=['Failed', 'Passed'])

    lr_2d = LogisticRegression(random_state=42)
    lr_2d.fit(X_top, y)

    DecisionBoundaryDisplay.from_estimator(
        lr_2d, X_top, response_method="predict",
        alpha=0.5, ax=axes[1],
        xlabel=f"{top_features[0]} (scaled)",
        ylabel=f"{top_features[1]} (scaled)",
    )

    scatter = axes[1].scatter(X_top[:, 0], X_top[:, 1], c=y, edgecolors='black',
                              s=50, cmap='coolwarm')
    axes[1].set_title(f'Logistic Regression Decision Boundary\n{top_features[0]} vs {top_features[1]}')
    axes[1].legend(handles=scatter.legend_elements()[0], labels=['Failed', 'Passed'])

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 60)

    print(f"\nRandom Forest Performance:")
    print(f"   Test Accuracy: {rf.score(X_test, y_test):.3f}")
    print(f"   ROC AUC Score: {roc_auc_rf:.3f}")

    print(f"\nLogistic Regression Performance:")
    print(f"   Test Accuracy: {lr.score(X_test, y_test):.3f}")
    print(f"   ROC AUC Score: {roc_auc_lr:.3f}")

    print(f"\ Top 2 Most Important Features:")
    for i, (feature, imp) in enumerate(zip(importance_df['feature'][::-1], importance_df['importance'][::-1])):
        print(f"   {i + 1}. {feature}: {imp:.3f}")

    print(f"\n Classification Report (Random Forest):")
    print(classification_report(y_test, y_pred_rf, target_names=['Failed', 'Passed']))

    print(f"\n SAMPLE PREDICTIONS:")
    print("-" * 40)

    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)

    for i, idx in enumerate(sample_indices):
        actual_label = "Passed" if y_test.iloc[idx] == 1 else "Failed"
        rf_prob = y_pred_proba_rf[idx]
        rf_pred = "Passed" if rf_prob > 0.5 else "Failed"
        lr_prob = y_pred_proba_lr[idx]
        lr_pred = "Passed" if lr_prob > 0.5 else "Failed"

        print(f"Student {i + 1}:")
        print(f"  Actual: {actual_label}")
        print(f"  RF Prediction: {rf_pred} (confidence: {rf_prob:.3f})")
        print(f"  LR Prediction: {lr_pred} (confidence: {lr_prob:.3f})")
        print()