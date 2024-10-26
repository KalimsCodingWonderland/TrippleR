from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.credentials import Credentials

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

# IBM Watsonx AI Credentials and Model Initialization
api_key = 'iwDOQ_4_8eOg_QH86FpoLxfCo7vXlUFb6_eGolQbgdnW'
url = 'https://us-south.ml.cloud.ibm.com/'
project_id = 'd0eaa248-e010-412c-8cf8-ba046b28f236'

credentials = Credentials(
    api_key=api_key,
    url=url
)

model_ai = Model(
    model_id=ModelTypes.LLAMA_2_13B_CHAT,
    credentials=credentials,
    project_id=project_id
)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', states=states, counties=counties,
                           disaster_types=disaster_types)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    state = data.get('state')
    county = data.get('county')
    disaster = data.get('disaster')

    # Create a DataFrame for prediction
    input_data = pd.DataFrame({
        'State': [state],
        'County': [county],
        'DisasterType': [disaster]
    })

    # Predict risk
    predicted_risk = model.predict(input_data)[0]
    predicted_risk = round(predicted_risk, 2)

    # Return the prediction as JSON for the JavaScript fetch call
    return jsonify({"risk_score": predicted_risk})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message')

    messages = [
        {"role": "system", "content": "You are an assistant that provides natural disaster preparedness recipes."},
        {"role": "user", "content": user_message}
    ]

    try:
        generated_response = model_ai.chat(messages=messages)
        response_content = generated_response['choices'][0]['message']['content']

        return jsonify({"response": response_content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

