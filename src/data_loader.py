import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

DATA_HOME = os.path.join(os.path.dirname(__file__), "data")

def load_and_split_data(test_size=0.2, random_state=42):
    data = fetch_california_housing(
        data_home=DATA_HOME,
        download_if_missing=False
    )

    X, y = data.data, data.target

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
