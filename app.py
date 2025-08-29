from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pickle
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'heart_disease_predictor_key'

# Load the trained model
print("Loading model...")
try:
    with open("model.pkl", "rb") as f:
        model_data = pickle.load(f)
    
    # Extract model components
    pipeline = model_data["pipeline"]
    features = model_data["features"]
    categorical_features = model_data.get("categorical_features", [])
    numerical_features = model_data.get("numerical_features", [])
    binary_features = model_data.get("binary_features", [])
    feature_mappings = model_data.get("feature_mappings", {})
    
    print(f"Model loaded successfully with {len(features)} features")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please run train_model.py first to create the model")
    pipeline = None
    features = []
    categorical_features = []
    numerical_features = []
    binary_features = []
    feature_mappings = {}

# Create directory for storing user predictions if it doesn't exist
if not os.path.exists('user_data'):
    os.makedirs('user_data')

# Function to map categorical values from form to model input
def map_categorical_values(form_data):
    mapped_data = {}
    
    # Map sex (from text to numeric)
    if 'sex' in form_data:
        sex_value = form_data['sex'].upper()
        if sex_value in ['M', 'MALE', '1']:
            mapped_data['sex'] = 1
        elif sex_value in ['F', 'FEMALE', '0']:
            mapped_data['sex'] = 0
    
    # Map chest pain type
    if 'chestpaintype' in form_data:
        cp_value = form_data['chestpaintype']
        cp_mapping = {
            '1': 'ATA',  # Atypical Angina
            '2': 'NAP',  # Non-Anginal Pain
            '3': 'ASY',  # Asymptomatic
            '4': 'TA'    # Typical Angina
        }
        mapped_data['chestpaintype'] = cp_mapping.get(cp_value, cp_value)
    
    # Map resting ECG
    if 'restingecg' in form_data:
        ecg_value = form_data['restingecg']
        ecg_mapping = {
            '0': 'Normal',
            '1': 'ST',
            '2': 'LVH'
        }
        mapped_data['restingecg'] = ecg_mapping.get(ecg_value, ecg_value)
    
    # Map exercise-induced angina
    if 'exerciseangina' in form_data:
        angina_value = form_data['exerciseangina']
        if angina_value in ['1', 'Y', 'YES', 'TRUE']:
            mapped_data['exerciseangina'] = 'Y'
        elif angina_value in ['0', 'N', 'NO', 'FALSE']:
            mapped_data['exerciseangina'] = 'N'
    
    # Map ST slope
    if 'st_slope' in form_data:
        slope_value = form_data['st_slope']
        slope_mapping = {
            '1': 'Up',
            '2': 'Flat',
            '3': 'Down'
        }
        mapped_data['st_slope'] = slope_mapping.get(slope_value, slope_value)
    
    # Map fasting blood sugar
    if 'fastingbs' in form_data:
        fbs_value = form_data['fastingbs']
        if fbs_value in ['1', 'Y', 'YES', 'TRUE']:
            mapped_data['fastingbs'] = 1
        elif fbs_value in ['0', 'N', 'NO', 'FALSE']:
            mapped_data['fastingbs'] = 0
    
    return mapped_data

@app.route("/")
def home():
    return render_template("splash.html")

@app.route("/predict_form")
def predict_form():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if pipeline is None:
        return render_template("index.html", error="Model not loaded. Please run train_model.py first.")
    
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Map categorical values
        mapped_values = map_categorical_values(form_data)
        form_data.update(mapped_values)
        
        # Create input data dictionary
        input_data = {}
        
        # Process numerical features
        for feature in numerical_features:
            if feature in form_data and form_data[feature]:
                input_data[feature] = float(form_data[feature])
            else:
                input_data[feature] = 0  # Default value if missing
        
        # Process categorical features
        for feature in categorical_features:
            if feature in form_data and form_data[feature]:
                input_data[feature] = form_data[feature]
            else:
                input_data[feature] = "Unknown"  # Default value if missing
        
        # Process binary features
        for feature in binary_features:
            if feature in form_data and form_data[feature]:
                input_data[feature] = int(form_data[feature])
            else:
                input_data[feature] = 0  # Default value if missing
        
        # Convert to DataFrame for prediction
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]  # Probability of class 1
        
        # Store user data and prediction
        user_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'input_data': input_data,
            'prediction': int(prediction),
            'probability': float(probability)
        }
        
        # Store in session for history
        if 'history' not in session:
            session['history'] = []
        
        session['history'].append(user_data)
        session['current_prediction'] = user_data
        
        # Save to file (optional)
        with open(f'user_data/prediction_{datetime.now().strftime("%Y%m%d%H%M%S")}.json', 'w') as f:
            json.dump(user_data, f)
        
        # Redirect to results page
        return redirect(url_for('results'))
    
    except Exception as e:
        error_message = f"Error during prediction: {str(e)}"
        print(error_message)
        return render_template("index.html", error=error_message)

@app.route("/results")
def results():
    if 'current_prediction' not in session:
        return redirect(url_for('predict_form'))
    
    prediction_data = session['current_prediction']
    prediction = prediction_data['prediction']
    probability = prediction_data['probability'] * 100  # Convert to percentage
    
    # Generate health advice based on prediction
    if prediction == 1:
        risk_status = "High Risk"
        advice = [
            "Consult with a healthcare professional as soon as possible",
            "Consider lifestyle changes such as improved diet and regular exercise",
            "Monitor your blood pressure and cholesterol levels regularly",
            "Reduce stress through meditation or other relaxation techniques",
            "Quit smoking if applicable"
        ]
    else:
        risk_status = "Low Risk"
        advice = [
            "Continue maintaining a healthy lifestyle",
            "Regular check-ups with your doctor are still important",
            "Stay physically active with at least 150 minutes of exercise per week",
            "Maintain a balanced diet rich in fruits, vegetables, and whole grains",
            "Limit alcohol consumption and avoid smoking"
        ]
    
    return render_template(
        "results.html",
        prediction=prediction,
        probability=probability,
        risk_status=risk_status,
        advice=advice,
        input_data=prediction_data['input_data']
    )

if __name__ == "__main__":
    app.run(debug=True)
