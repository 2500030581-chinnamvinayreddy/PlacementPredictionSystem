from flask import Flask, render_template

from src.data.load_data import load_data, get_summary

app = Flask(__name__)


@app.route("/")
def home():

    return render_template("home.html")


@app.route("/dataset")
def dataset():

    # Load dataset
    df = load_data()

    # Get summary
    summary = get_summary(df)

    return render_template(
        "load_dataset.html",
        rows=summary["rows"],
        cols=summary["columns"],
        target=summary["target"],
        columns=df.columns.tolist(),
        data=df.head(10).values.tolist()
    )


@app.route("/eda")
def eda():

    return render_template("eda.html")


@app.route("/preprocessing")
def preprocessing():

    # Load dataset
    df = load_data()

    return render_template(
        "preprocessing.html",
        rows=df.shape[0],
        cols=df.shape[1]
    )


@app.route("/models")
def models():

    return render_template("models.html")


@app.route("/evaluation")
def evaluation():

    return render_template("evaluation.html")


@app.route("/comparison")
def comparison():

    return render_template("comparison.html")


@app.route("/prediction")
def prediction():

    return render_template("prediction.html")


if __name__ == "__main__":
    app.run(debug=True)