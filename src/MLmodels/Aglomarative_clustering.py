from src.data.load_data import load_data
from src.data.preprocess import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

def create_model(k):
    model = AgglomerativeClustering(n_clusters=k,
                                    linkage='ward')
    return model

def train_model(model, X):
    model.fit(X)
    print("Agglomerative Clustering Result:")
    return model

def evaluate_model(model, X):
    labels = model.labels_
    score = silhouette_score(X, labels)
    print("Silhouette Result:")
    print(score)
    print(labels)

def display_dendogram(X, cut_distance):
    linked = linkage(X, method="ward")
    plt.figure(figsize=(12,6))
    dendrogram(linked,truncate_mode='lastp',p=30)
    plt.axhline(y=cut_distance,linestyle='--')
    plt.title('Dendogram Plot')


def main():
    df = load_data()
    print("Data Shape:")
    print(df.shape)

    df = df.sample(n=500,random_state=42)

    x = split_x_data(
        df, drop_columns=['StudentID', 'PlacementStatus', 'IsAnomaly', 'Salary Package']
    )

    numerical_features, categorical_features = identify_features(x)
    print("Numerical features:", numerical_features)
    print("Categorical features:", categorical_features)

    one_hot_features = ['Gender', 'City', 'Stream', 'Specialisation', 'Hostel', 'HistoryOfBacklogs']
    ordinal_features = ['CollegeTier', 'CGPA_Tier']

    x, _, imputer = handle_missing_values(x, x, numerical_features)
    print("Missing values handled.")

    x, _, scaler = standardize_data(x, x, numerical_features)
    print("Standardization completed.")

    x, _, one_hot_encoder = one_hot_encode_data(x, x, one_hot_features)
    print("One-hot encoding completed.")

    x, _, ordinal_encoder = ordinal_encode_data(x, x, ordinal_features)
    print("Ordinal encoding completed.")

    cut_distance = 20

    linked = linkage(x, method="ward")
    clusters = fcluster(linked, t=cut_distance, criterion='distance')
    k = len(set(clusters))
    print("Clusters at cut_distance =", cut_distance, "->", k)

    model = create_model(k)
    model = train_model(model, x)
    evaluate_model(model, x)

    display_dendogram(x, cut_distance)
    plt.show()

if __name__ == "__main__":
    main()