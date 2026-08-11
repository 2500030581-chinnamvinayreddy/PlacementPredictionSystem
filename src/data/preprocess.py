from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.data.load_data import load_data

def split_data(df):
    x = df.drop(columns=["PlacementStatus"])
    y = df["PlacementStatus"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return x_train, x_test, y_train, y_test

def identify_features(x):
    numerical_features = x.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_features = x.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    return numerical_features, categorical_features

def standardize_data(x_train, x_test, numerical_features):
    scaler = StandardScaler()
    x_train = x_train.copy()
    x_test = x_test.copy()
    x_train[numerical_features] = scaler.fit_transform(
        x_train[numerical_features]
    )
    return x_train, x_test, scaler

if __name__ == "__main__":

    df = load_data()

    x_train, x_test, y_train, y_test = split_data(df)

    numerical_features, categorical_features = identify_features(x_train)

    print("Numerical Features:")
    print(numerical_features)

    print("Categorical Features:")
    print(categorical_features)

    x_train, x_test, scaler = standardize_data(
        x_train,
        x_test,
        numerical_features
    )

    print("Training Data:")
    print(x_train)

    print("Testing Data:")
    print(x_test)