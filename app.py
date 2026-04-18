from flask import Flask, render_template, request
from src.predict import predict_score

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        input_data = {
            "school": request.form["school"],
            "sex": request.form["sex"],
            "age": int(request.form["age"]),
            "address": request.form["address"],
            "famsize": request.form["famsize"],
            "Pstatus": request.form["Pstatus"],
            "Medu": int(request.form["Medu"]),
            "Fedu": int(request.form["Fedu"]),
            "Mjob": request.form["Mjob"],
            "Fjob": request.form["Fjob"],
            "reason": request.form["reason"],
            "guardian": request.form["guardian"],
            "traveltime": int(request.form["traveltime"]),
            "studytime": int(request.form["studytime"]),
            "failures": int(request.form["failures"]),
            "schoolsup": request.form["schoolsup"],
            "famsup": request.form["famsup"],
            "paid": request.form["paid"],
            "activities": request.form["activities"],
            "nursery": request.form["nursery"],
            "higher": request.form["higher"],
            "internet": request.form["internet"],
            "romantic": request.form["romantic"],
            "famrel": int(request.form["famrel"]),
            "freetime": int(request.form["freetime"]),
            "goout": int(request.form["goout"]),
            "Dalc": int(request.form["Dalc"]),
            "Walc": int(request.form["Walc"]),
            "health": int(request.form["health"]),
            "absences": int(request.form["absences"]),
            "G1": int(request.form["G1"]),
            "G2": int(request.form["G2"]),
            "course": request.form["course"]
        }

        prediction = predict_score(input_data)

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)