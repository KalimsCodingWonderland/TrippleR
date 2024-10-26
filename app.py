from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('risk_model.joblib')

# Load unique states, counties, and disaster types for dropdowns
data = pd.read_csv('PredictionDataSet.csv')
disaster_types = ['Avalanche', 'Coastal Flooding', 'Cold Wave', 'Drought', 'Earthquake',
                  'Hail', 'Heatwave', 'Hurricane', 'Icestorm', 'Landslide', 'Lightning',
                  'Riverine', 'Flooding', 'Strong Wind', 'Tornado', 'Tsunami',
                  'Volcanic Activity', 'Wildfire', 'Winter Weather']

states = sorted(data['State'].unique())
counties = sorted(data['County'].unique())

# Define annual risk increase rate (e.g., 1.5%)
ANNUAL_RISK_INCREASE = 0.015


def get_future_risk_scores(base_risk, max_years=50):
    """
    Generate future risk scores over a range of years based on an annual increase rate.
    """
    years = np.arange(0, max_years + 1)
    risk_scores = base_risk * (1 + ANNUAL_RISK_INCREASE) ** years
    risk_scores = np.round(risk_scores, 2)
    return {'labels': years.tolist(), 'data': risk_scores.tolist()}


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        state = request.form['state']
        county = request.form['county']
        disaster = request.form['disaster']
        time = int(request.form.get('time', 0))

        # Create a DataFrame for prediction
        input_data = pd.DataFrame({
            'State': [state],
            'County': [county],
            'DisasterType': [disaster]
            # 'Year' is not used in the model; it's for post-prediction adjustment
        })

        # Predict base risk
        base_risk = model.predict(input_data)[0]
        base_risk = round(base_risk, 2)

        # Adjust risk based on time
        adjusted_risk = base_risk * (1 + ANNUAL_RISK_INCREASE) ** time
        adjusted_risk = round(adjusted_risk, 2)

        # Generate chart data
        chart_data = get_future_risk_scores(base_risk, max_years=50)

        return render_template('index.html',
                               states=states,
                               counties=counties,
                               disaster_types=disaster_types,
                               prediction=adjusted_risk,
                               base_prediction=base_risk,
                               state=state,
                               county=county,
                               disaster=disaster,
                               time=time,
                               chart_data=chart_data)
    return render_template('index.html',
                           states=states,
                           counties=counties,
                           disaster_types=disaster_types,
                           prediction=None)


if __name__ == '__main__':
    app.run(debug=True)
