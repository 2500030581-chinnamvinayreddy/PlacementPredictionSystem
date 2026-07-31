import pandas as pd
def load_data():
    df=pd.read_csv(r"C:\Users\ADMIN\PycharmProjects\PlacementPredictionSystem\data\placement_data (1).csv")
    return df
def get_summary(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "target":"PlacementStatus"
    }
if __name__ == "__main__":
    df=load_data()
    summary=(get_summary(df))
    print(summary)

    print(df.head())