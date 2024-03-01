from flask import Flask, render_template, request
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
lr = pickle.load(open('lr.pkl', 'rb'))
std = pickle.load(open('std.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from the form
    val1 = float(request.form['bedrooms'])
    val2 = float(request.form['bathrooms'])
    val3 = float(request.form['sqft_living'])
    val4 = float(request.form['floors'])
    val5 = float(request.form['view'])
    val6 = float(request.form['sqft_above'])
    val7 = float(request.form['sqft_basement'])
    val8 = float(request.form['yr_built'])
    val9 = float(request.form['zipcode'])
    val10 = float(request.form['lat'])
    val11 = float(request.form['long'])

    # Create a 2D array with the input values
    new_house = np.array([[val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11]])

    # Check if the scaler is fitted
    if not hasattr(std, 'mean_'):
        raise ValueError("Scaler is not fitted. Make sure to fit the scaler before using it.")

    # Standardize the input data using the pre-trained scaler
    standardized_input = std.transform(new_house)

    # Predict using the model
    data = lr.predict(standardized_input)

    return render_template('index.html', data=abs(data[0]))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
