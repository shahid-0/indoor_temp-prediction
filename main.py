from flask import Flask, request, render_template, url_for
import joblib
import numpy as np


app = Flask(__name__)
model = joblib.load('model.pkl')
# app.config['DEBUG'] = True

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    int_features = [float(x) for x in request.form.values()]
    print(int_features)
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    outputs = prediction[0].tolist()
    return render_template("home.html", temp_predict=f"Indoor Temp: {outputs[0]}", humidity_predict=f"Indoor Humidity: {outputs[1]}")

if "__name__" == "__main__":
    app.run()