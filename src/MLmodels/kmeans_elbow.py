import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pathlib import Path


def main():

    # Get project root directory
    BASE_DIR = Path(__file__).resolve().parents[2]

    # Automatically find placement_data.csv
    matches = list(BASE_DIR.rglob("placement_data.csv"))

    if not matches:
        print("ERROR: placement_data.csv was not found anywhere inside:")
        print(BASE_DIR)
        return

    DATA_PATH = matches[0]

    print("Loading dataset from:")
    print(DATA_PATH)

    # Load dataset
    data = pd.read_csv(DATA_PATH)

    print("\nDataset loaded successfully!")
    print("Shape:", data.shape)

    # Check required columns
    required_columns = ["CGPA", "AptitudeTestScore"]

    for column in required_columns:
        if column not in data.columns:
            print(f"\nERROR: Column '{column}' not found in dataset.")
            print("\nAvailable columns:")
            print(list(data.columns))
            return

    # Select features
    features = data[required_columns].dropna()

    print("\nSelected Features:")
    print(features.head())

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # WCSS
    wcss = []

    # Silhouette scores
    silhouette_scores = []

    k_values = range(2, 11)

    for k in k_values:

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(X)

        wcss.append(kmeans.inertia_)

        silhouette_scores.append(
            silhouette_score(X, labels)
        )

    # Print results
    print("\nK-Means Results")
    print("-" * 50)

    for k, wcss_value, score in zip(
        k_values,
        wcss,
        silhouette_scores
    ):
        print(
            f"K = {k} | "
            f"WCSS = {wcss_value:.2f} | "
            f"Silhouette Score = {score:.4f}"
        )

    # Best K
    best_k = k_values[
        silhouette_scores.index(
            max(silhouette_scores)
        )
    ]

    print("\nRecommended K:", best_k)

    # Elbow Method
    plt.figure(figsize=(8, 5))

    plt.plot(
        k_values,
        wcss,
        marker="o"
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS")
    plt.title("K-Means Elbow Method")

    plt.xticks(list(k_values))
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Silhouette Score
    plt.figure(figsize=(8, 5))

    plt.plot(
        k_values,
        silhouette_scores,
        marker="o"
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("K-Means Silhouette Score")

    plt.xticks(list(k_values))
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()