import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, mean_squared_error
import numpy as np


def load_preprocessed_data():
    train_path = r"C:\Users\ADMIN\PycharmProjects\PlacementPredictionSystem\data\preprocessed_test1.csv"
    test_path = r"C:\Users\ADMIN\PycharmProjects\PlacementPredictionSystem\data\preprocessed_test2.csv"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    return train_data, test_data


def split_features_target(train_data, test_data):
    X_train = train_data.drop(columns=["PlacementStatus"])
    y_train = train_data["PlacementStatus"]

    X_test = test_data.drop(columns=["PlacementStatus"])
    y_test = test_data["PlacementStatus"]

    return X_train, X_test, y_train, y_test


def create_model():
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    return model


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\nAccuracy:")
    print(model.score(X_test, y_test))

    mse = mean_squared_error(y_test, y_pred)
    print("\nMean Squared Error (MSE):")
    print(mse)

    rmse = np.sqrt(mse)
    print("\nRoot Mean Squared Error (RMSE):")
    print(rmse)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


def save_model(model):
    model_path = r"C:\Users\ADMIN\PycharmProjects\PlacementPredictionSystem\modelslogistic_regression.pkl"

    joblib.dump(model, model_path)

    print("\nModel Successfully saved:")
    print(model_path)


if __name__ == "__main__":

    train_data, test_data = load_preprocessed_data()

    print("Training Data Shape:")
    print(train_data.shape)

    print("\nTesting Data Shape:")
    print(test_data.shape)

    X_train, X_test, y_train, y_test = split_features_target(
        train_data,
        test_data
    )

    print("\nX_train Shape:")
    print(X_train.shape)

    print("\nX_test Shape:")
    print(X_test.shape)

    print("\ny_train Shape:")
    print(y_train.shape)

    print("\ny_test Shape:")
    print(y_test.shape)


    model = create_model()


    model = train_model(
        model,
        X_train,
        y_train
    )

    print("\nLogistic Regression Training Completed!")


    evaluate_model(
        model,
        X_test,
        y_test
    )


    save_model(model)