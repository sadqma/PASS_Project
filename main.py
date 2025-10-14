from src.data_preprocessing import load_data
from src.visualizations import plot_correlations, plot_feature_vs_success
from src.model_training import train_baseline_model

def main():
    df = load_data()
    plot_correlations(df)
    plot_feature_vs_success(df, "attendance")
    train_baseline_model(df)

if __name__ == "__main__":
    main()
fdfd'ss
ws
set