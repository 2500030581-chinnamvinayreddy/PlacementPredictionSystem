import pandas as pd
def load_iris():
    df=pd.read_csv(r"C:\Users\ADMIN\PycharmProjects\PlacementPredictionSystem\data\iris.csv")
    return df
def get_summary(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "target":"PlacementStatus"
    }
if __name__ == "__main__":
    df=load_iris()
    summary=(get_summary(df))
    print(summary)

    print(df.head())