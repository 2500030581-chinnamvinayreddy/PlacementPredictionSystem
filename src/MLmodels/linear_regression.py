from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.data.load_data import load_data

from src.data.preprocess import (
    split_data,
    identify_features,
    handle_missing_values,
    standardize_data,
    one_hot_encode_data,
    ordinal_encode_data
)

def create_models():

    models = {
        "Linear Regression": LinearRegression(),

        "Ridge Regression": Ridge(
            alpha=1.0
        ),

        "Lasso Regression": Lasso(
            alpha=0.01
        ),

        "ElasticNet Regression": ElasticNet(
            alpha=0.01,
            l1_ratio=0.5
        )
    }

    return models

def train_model(model, x_train, y_train):

    model.fit(x_train, y_train)

    return model

def predict(model, x_test):

    y_pred = model.predict(x_test)

    return y_pred

def evaluate(y_test, y_pred):

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        y_pred
    )

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

def main():

    df = load_data()

    print("========================================")
    print("DATASET INFORMATION")
    print("========================================")

    print("\nOriginal Dataset Shape:")
    print(df.shape)

    x_train, x_test, y_train, y_test = split_data(
        df,
        target_column="Salary Package",
        drop_columns=[
            "StudentID",
            "PlacementStatus",
            "IsAnamoly"
        ]
    )

    print("\nTraining Shape:")
    print(x_train.shape)

    print("\nTesting Shape:")
    print(x_test.shape)

    print("\nTraining Target Shape:")
    print(y_train.shape)

    print("\nTesting Target Shape:")
    print(y_test.shape)

    numerical_features, categorical_features = identify_features(
        x_train
    )

    print("\n========================================")
    print("FEATURE INFORMATION")
    print("========================================")

    print("\nNumerical Features:")

    for feature in numerical_features:
        print(" -", feature)

    print("\nNumber of Numerical Features:")
    print(len(numerical_features))


    print("\nCategorical Features:")

    for feature in categorical_features:
        print(" -", feature)

    print("\nNumber of Categorical Features:")
    print(len(categorical_features))

    one_hot_features = [
        "Gender",
        "City",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs"
    ]

    ordinal_features = [
        "CollegeTier",
        "CGPA_Tier"
    ]

    x_train, x_test, imputer = handle_missing_values(
        x_train,
        x_test,
        numerical_features
    )

    print("\nMissing Value Handling Completed")

    x_train, x_test, scaler = standardize_data(
        x_train,
        x_test,
        numerical_features
    )

    print("\nStandardization Completed")

    x_train, x_test, one_hot_encoder = one_hot_encode_data(
        x_train,
        x_test,
        one_hot_features
    )

    print("\nOne-Hot Encoding Completed")

    x_train, x_test, ordinal_encoder = ordinal_encode_data(
        x_train,
        x_test,
        ordinal_features
    )

    print("\nOrdinal Encoding Completed")

    print("\n========================================")
    print("FINAL DATA")
    print("========================================")

    print("\nFinal X_train Shape:")
    print(x_train.shape)

    print("\nFinal X_test Shape:")
    print(x_test.shape)

    models = create_models()

    print("\n========================================")
    print("REGRESSION MODELS")
    print("========================================")

    results = {}

    for name, model in models.items():

        print("\n----------------------------------------")
        print(name)
        print("----------------------------------------")

        # Train
        model = train_model(
            model,
            x_train,
            y_train
        )

        # Predict
        y_pred = predict(
            model,
            x_test
        )

        # Evaluate
        metrics = evaluate(
            y_test,
            y_pred
        )

        results[name] = metrics

        print("\nMAE :", metrics["MAE"])
        print("MSE :", metrics["MSE"])
        print("RMSE:", metrics["RMSE"])
        print("R2  :", metrics["R2"])

    print("\n========================================")
    print("MODEL COMPARISON")
    print("========================================")

    for name, metrics in results.items():

        print("\n", name)

        print(
            "MAE :",
            round(metrics["MAE"], 4)
        )

        print(
            "MSE :",
            round(metrics["MSE"], 4)
        )

        print(
            "RMSE:",
            round(metrics["RMSE"], 4)
        )

        print(
            "R2  :",
            round(metrics["R2"], 4)
        )

if __name__ == "__main__":
    main()