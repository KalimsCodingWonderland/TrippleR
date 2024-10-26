from flask import Flask, request, render_template
import joblib
import pandas as pd

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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        state = request.form['state']
        county = request.form['county']
        disaster = request.form['disaster']

        # Create a DataFrame for prediction
        input_data = pd.DataFrame({
            'State': [state],
            'County': [county],
            'DisasterType': [disaster]
        })

        # Predict risk
        predicted_risk = model.predict(input_data)[0]
        predicted_risk = round(predicted_risk, 2)

        return render_template('index.html', states=states, counties=counties,
                               disaster_types=disaster_types,
                               prediction=predicted_risk,
                               state=state, county=county, disaster=disaster)
    return render_template('index.html', states=states, counties=counties,
                           disaster_types=disaster_types, prediction=None)

if __name__ == '__main__':
    app.run(debug=True)