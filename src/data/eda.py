from src.data.load_data import load_data
import matplotlib.pyplot as plt
import seaborn as sns
def basic_eda(df):
    print("First five rows")
    print(df.head())
    print("Last five rows")
    print(df.tail())
    print("Rows 25 to 35")
    print(df.iloc[24:35])
    print("Sample of 10 records")
    print(df.sample(10))
    print("Column names")
    print(df.columns)
    print("=" * 50)
    print("Datatypes")
    print(df.dtypes)
    print("Complete Information")
    print(df.info())
    missing=df.isnull().sum()
    print(missing[missing>0])
    print("Duplicated values")
    print(df.duplicated().sum())
    print(df["PlacementStatus"].value_counts())
    count=df["PlacementStatus"].value_counts()
    plt.figure(figsize=(6,5))
    plt.bar(count.index,count.values)
    plt.title("Distribution of Placement Status")
    plt.xlabel("Placement Status")
    plt.ylabel("Count")
    plt.savefig(r"C:\Users\ADMIN\PycharmProjects\PlacementPredictionSystem\results\placement_status.png")
    plt.show()

def univariate(df):
    plt.figure(figsize=(6,5))
    plt.hist(df["CGPA"],bins=10,edgecolor="black")
    plt.title("Histogram of CGPA")
    plt.xlabel("CGPA")
    plt.ylabel("Frequency")
    plt.savefig(r"/app/static\charts\histogram.png")
    plt.show()

    gendercount=df["Gender"].value_counts()
    plt.figure(figsize=(6,5))
    plt.pie(gendercount,labels= gendercount.index,autopct="%1.1f%%",startangle=90)
    plt.title("Gender Distribution Piechart")
    plt.savefig(r"/app/static\charts\gender.png")
    plt.show()

def bivariate(df):
    plt.figure(figsize=(6,5))
    plt.scatter(df["CGPA"],df["AptitudeTestScore"],c="green")
    plt.title("CGPA vs Aptitude Test Score")
    plt.xlabel("CGPA")
    plt.ylabel("Aptitude Test Score")
    plt.savefig(r"/app/static\charts\cgpa_aptitudescore_scatter")
    plt.show()

    placed=df[df["PlacementStatus"]==1]["CGPA"]
    not_placed=df[df["PlacementStatus"]==0]["CGPA"]
    plt.boxplot([placed,not_placed],label=["placed","not_placed"])
    plt.title("CGPA vs PlacementStatus")
    plt.xlabel("PlacementStatus")
    plt.ylabel("CGPA")
    plt.savefig(r"C:\Users\ADMIN\PycharmProjects\PlacementPredictionSystem\results\placementStatus.png")
    plt.show()

def multivariate(df):
    data = df[["CGPA","AptitudeTestScore","PlacementStatus"]]
    correlation=df.corr(numeric_only=True)
    plt.figure(figsize=(8,6))
    sns.heatmap(correlation,
                annot=True,
                cmap="coolwarm",
                fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(r"/app/static\charts\correlation_matrix_1.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    df = load_data()
    #basic_eda(df)
    #univariate(df)
    #bivariate(df)
    multivariate(df)