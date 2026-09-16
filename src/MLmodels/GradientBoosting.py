import pandas as pd
import matplotlib.pyplot as plt

from src.data.load_data import load_data
from src.data.preprocess import (
    split_data,
    identify_features,
    handle_missing_values,
    one_hot_encode_data,
    ordinal_encode_data
)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def create_model():

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    return model


def train_model(model, x_train, y_train):

    model.fit(x_train, y_train)

    print("\nGradient Boosting trained successfully.")

    return model


def evaluate_model(model, x_test, y_test):

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\n========================================")
    print("MODEL EVALUATION")
    print("========================================")

    print("\nAccuracy:")
    print(accuracy)

    print("\nAccuracy Percentage:")
    print(round(accuracy * 100, 2), "%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Not Placed",
                "Placed"
            ]
        )
    )

    print("\nConfusion Matrix:")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(cm)

    return y_pred


def display_confusion_matrix(
        y_test,
        y_pred
):

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(8, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Not Placed",
            "Placed"
        ]
    )

    display.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title(
        "Gradient Boosting - Confusion Matrix"
    )

    plt.tight_layout()

    plt.show()


def display_feature_importance(
        model,
        feature_names
):

    importance = model.feature_importances_

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    feature_importance = (
        feature_importance
        .sort_values(
            by="Importance",
            ascending=False
        )
    )

    print("\n========================================")
    print("FEATURE IMPORTANCE")
    print("========================================")

    print(
        feature_importance.to_string(
            index=False
        )
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        feature_importance["Feature"],
        feature_importance["Importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.title(
        "Gradient Boosting Feature Importance"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.show()


def main():

    df = load_data()

    print("========================================")
    print("DATASET INFORMATION")
    print("========================================")

    print("\nOriginal Dataset Shape:")
    print(df.shape)

    x_train, x_test, y_train, y_test = split_data(
        df,
        target_column="PlacementStatus",
        drop_columns=[
            "StudentID",
            "Salary Package",
            "IsAnomaly"
        ]
    )

    print("\nTraining Data Shape:")
    print(x_train.shape)

    print("\nTesting Data Shape:")
    print(x_test.shape)

    print("\nTraining Target Shape:")
    print(y_train.shape)

    print("\nTesting Target Shape:")
    print(y_test.shape)

    # ========================================
    # IDENTIFY FEATURES
    # ========================================

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

    # ========================================
    # ENCODING
    # ========================================

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

    # ========================================
    # MISSING VALUE HANDLING
    # ========================================

    x_train, x_test, imputer = handle_missing_values(
        x_train,
        x_test,
        numerical_features
    )

    print("\n========================================")
    print("MISSING VALUE HANDLING")
    print("========================================")

    print("\nMissing Values Handling Completed.")

    # ========================================
    # ONE-HOT ENCODING
    # ========================================

    x_train, x_test, one_hot_encoded = one_hot_encode_data(
        x_train,
        x_test,
        one_hot_features
    )

    print("\n========================================")
    print("ONE-HOT ENCODING")
    print("========================================")

    print("\nOne-Hot Encoding Completed.")

    # ========================================
    # ORDINAL ENCODING
    # ========================================

    x_train, x_test, ordinal_encoded = ordinal_encode_data(
        x_train,
        x_test,
        ordinal_features
    )

    print("\n========================================")
    print("ORDINAL ENCODING")
    print("========================================")

    print("\nOrdinal Encoding Completed.")

    # ========================================
    # FINAL DATA
    # ========================================

    print("\n========================================")
    print("FINAL DATA")
    print("========================================")

    print("\nFinal X_train Shape:")
    print(x_train.shape)

    print("\nFinal X_test Shape:")
    print(x_test.shape)

    print("\nFinal Features:")

    print(list(x_train.columns))

    # ========================================
    # CREATE MODEL
    # ========================================

    model = create_model()

    print("\n========================================")
    print("MODEL")
    print("========================================")

    print("\nGradient Boosting Parameters:")

    print(
        "Number of Estimators:",
        model.n_estimators
    )

    print(
        "Learning Rate:",
        model.learning_rate
    )

    print(
        "Max Depth:",
        model.max_depth
    )

    print(
        "Random State:",
        model.random_state
    )

    # ========================================
    # TRAIN MODEL
    # ========================================

    model = train_model(
        model,
        x_train,
        y_train
    )

    # ========================================
    # EVALUATE MODEL
    # ========================================

    y_pred = evaluate_model(
        model,
        x_test,
        y_test
    )

    # ========================================
    # CONFUSION MATRIX
    # ========================================

    display_confusion_matrix(
        y_test,
        y_pred
    )

    # ========================================
    # FEATURE IMPORTANCE
    # ========================================

    display_feature_importance(
        model,
        x_train.columns
    )


if __name__ == "__main__":
    main()