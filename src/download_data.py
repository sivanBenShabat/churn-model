import urllib.request
from pathlib import Path

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "telco_churn.csv"


def download_data():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, OUTPUT_PATH)
    print(f"Saved dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    download_data()
