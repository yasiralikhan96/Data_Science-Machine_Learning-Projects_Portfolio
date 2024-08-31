import _sqlite3
import numpy as np
import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", "rb"))

@app.route('/', methods=['GET', 'POST'])
def index():
    locations = sorted(data['location'].unique())
    prediction_result = None

    if request.method == 'POST':
        # Process form data here
        location = request.form['location']
        bhk = request.form['bhk']
        bathrooms = request.form['bathrooms']
        sqft = request.form['sqft']

        print(location, bhk, bathrooms, sqft)
        input_df = pd.DataFrame([[location, sqft, bathrooms, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
        prediction = pipe.predict(input_df)[0]

        prediction_result = f"The predicted house price is: {prediction:.2f} Lacs"

    return render_template('index.html', locations=locations, prediction_result=prediction_result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
