from src.data.load_iris import load_iris
import matplotlib.pyplot as plt

def basic_eda1(df):
    print(df["species"].value_counts())

    count = df["species"].value_counts()

    plt.figure(figsize=(6,5))
    plt.bar(count.index, count.values)

    plt.title("Distribution of Species")
    plt.xlabel("Species")
    plt.ylabel("Count")

    plt.savefig(r"C:\Users\ADMIN\PycharmProjects\PlacementPredictionSystem\results\iris_distribution.png")
    plt.show()


if __name__ == "__main__":
    df = load_iris()
    basic_eda1(df)