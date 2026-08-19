from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "telco_churn.csv"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"


def run_eda():
    df = pd.read_csv(DATA_PATH)

    print("Shape:", df.shape)
    print("\nColumn dtypes:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())

    churn_rate = (df["Churn"] == "Yes").mean()
    print(f"\nChurn rate: {churn_rate:.2%}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    churn_counts = df["Churn"].value_counts()
    plt.figure(figsize=(6, 4))
    churn_counts.plot(kind="bar", color=["steelblue", "indianred"])
    plt.title("Class Balance: Churn")
    plt.xlabel("Churn")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "churn_class_balance.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    df[df["Churn"] == "No"]["tenure"].plot(kind="hist", bins=30, alpha=0.6, label="No", color="steelblue")
    df[df["Churn"] == "Yes"]["tenure"].plot(kind="hist", bins=30, alpha=0.6, label="Yes", color="indianred")
    plt.title("Tenure Distribution by Churn")
    plt.xlabel("Tenure (months)")
    plt.ylabel("Frequency")
    plt.legend(title="Churn")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "tenure_by_churn.png")
    plt.close()

    print(f"\nPlots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    run_eda()
