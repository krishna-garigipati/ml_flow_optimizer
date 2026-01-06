from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def load_and_split_data(test_size=0.2, random_state=42):
    data = fetch_california_housing()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test
