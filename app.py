from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('risk_model.joblib')
except FileNotFoundError:
    raise FileNotFoundError("The model file 'risk_model.joblib' was not found. Ensure it's in the correct directory.")

# Load unique states, counties, and disaster types for dropdowns
try:
    data = pd.read_csv('PredictionDataSet.csv')
except FileNotFoundError:
    raise FileNotFoundError("The data file 'PredictionDataSet.csv' was not found. Ensure it's in the correct directory.")

disaster_types = ['Avalanche', 'Coastal Flooding', 'Cold Wave', 'Drought', 'Earthquake',
                  'Hail', 'Heatwave', 'Hurricane', 'Icestorm', 'Landslide', 'Lightning',
                  'Riverine', 'Flooding', 'Strong Wind', 'Tornado', 'Tsunami',
                  'Volcanic Activity', 'Wildfire', 'Winter Weather']

states = sorted(data['State'].unique())
counties = sorted(data['County'].unique())

# Define annual risk increase rate (e.g., 1.5%)
ANNUAL_RISK_INCREASE = 0.015


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html',
                           states=states,
                           counties=counties,
                           disaster_types=disaster_types,
                           prediction=None)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle AJAX POST requests to calculate the base risk score.
    Expects JSON data with 'state', 'county', and 'disaster'.
    Returns JSON response with 'risk_score'.
    """
    if request.is_json:
        data = request.get_json()
        state = data.get('state')
        county = data.get('county')
        disaster = data.get('disaster')

        # Validate input
        if not all([state, county, disaster]):
            return jsonify({'error': 'Missing data: state, county, and disaster are required.'}), 400

        # Create a DataFrame for prediction
        input_data = pd.DataFrame({
            'State': [state],
            'County': [county],
            'DisasterType': [disaster]
            # 'Year' is not used in the model; it's for post-prediction adjustment
        })

        # Predict base risk
        try:
            base_risk = model.predict(input_data)[0]
            base_risk = round(base_risk, 2)
        except Exception as e:
            return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

        return jsonify({'risk_score': base_risk})
    else:
        return jsonify({'error': 'Request must be in JSON format.'}), 400


@app.route('/favicon.ico')
def favicon():
    """
    Serve the favicon.ico file to eliminate 404 errors for favicon requests.
    Ensure that 'favicon.ico' is placed inside the 'static' directory.
    """
    return app.send_static_file('favicon.ico')


if __name__ == '__main__':
    # Determine the port to run the app on
    port = int(os.environ.get('PORT', 5000))
    # Run the Flask app
    app.run(debug=True, port=port)