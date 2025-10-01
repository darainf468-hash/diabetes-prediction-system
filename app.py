from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import tensorflow as tf
import joblib
import traceback

# Initialize the Flask application
app = Flask(__name__, static_folder='static', template_folder='templates')

# --- LOAD YOUR NEW MODEL AND SCALER ---
# This section has the correct try/except structure.
try:
    print("--- Loading the final improved model and scaler... ---")
    model = tf.keras.models.load_model("final_diabetes_model.keras")
    scaler = joblib.load("final_scaler.pkl")
    print("--- Model and Scaler Loaded Successfully ---")
except Exception as e:
    print(f"!!! CRITICAL ERROR: Could not load model or scaler: {e} !!!")
    model = None
    scaler = None

# --- Feature names in the correct order for the model ---
feature_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# --- Recommender System Logic ---
def generate_recommendations(prediction_result, patient_data_dict):
    recommendations = []
    data = patient_data_dict

    if prediction_result == 1:
        recommendations.append("ACTION REQUIRED: The model predicts a high likelihood of diabetes.")
        if data['Glucose'] > 126: recommendations.append("- High Glucose: Consult a doctor immediately. It is critical to reduce sugar and refined carb intake.")
        if data['BMI'] >= 30: recommendations.append("- Obesity: Focus on a structured weight-loss plan with both diet and exercise. Consult a nutritionist.")
        elif data['BMI'] >= 25: recommendations.append("- Overweight: Increase daily physical activity and control food portions to manage weight.")
    else:
        recommendations.append("PREVENTION FOCUS: The model predicts a low likelihood of diabetes. The following are preventative recommendations based on your risk factors:")
        risk_found = False
        if data['Glucose'] > 100: recommendations.append("- Borderline Glucose: You are at risk. Reduce sugar intake and get regular check-ups."); risk_found = True
        if data['BMI'] >= 25: recommendations.append("- High BMI: Start a weight management plan and increase daily activity to reduce future risk."); risk_found = True
        if data['Age'] > 45: recommendations.append("- Age Factor: As you are over 45, regular annual health screenings are highly recommended."); risk_found = True
        if not risk_found: recommendations.append("- Excellent Health Profile: Continue maintaining your healthy diet and active lifestyle.")
            
    return recommendations

# --- Define Routes for Each Page ---
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/recommendations')
def recommendations_page():
    return render_template('recommendations.html')

@app.route('/contact')
def contact_page():
    return render_template('contact.html')

@app.route('/about_diabetes')
def about_diabetes_page():
    return render_template('about_diabetes.html')

@app.route('/about_us')
def about_us_page():
    return render_template('about_us.html')


# --- API Endpoint for Prediction ---
@app.route("/predict_api", methods=["POST"])
def predict_api():
    if not model or not scaler:
        return jsonify({'error': 'Model or scaler is not available. Check server logs.'}), 500

    try:
        data_from_form = request.get_json()
        
        # Create a dictionary and get values safely, ensuring they are floats
        patient_data_dict = {name: float(data_from_form.get(name.lower(), 0)) for name in feature_names}
        
        # Create a list of the feature values in the correct order
        features_list = [patient_data_dict[name] for name in feature_names]
        
        # Preprocess input
        features_array = np.array(features_list).reshape(1, -1)
        scaled_features = scaler.transform(features_array)
        reshaped_features = scaled_features.reshape((1, 1, 8))
        
        # Make a prediction
        prediction_proba = model.predict(reshaped_features, verbose=0)
        prediction_class = 1 if prediction_proba[0][0] > 0.5 else 0
        
        # Generate recommendations
        recommendations = generate_recommendations(prediction_class, patient_data_dict)

        # Send the results back
        return jsonify({
            "prediction": prediction_class,
            "probability": f"{prediction_proba[0][0]:.2f}",
            "recommendations": recommendations,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "An error occurred on the server.", "details": str(e)}), 500
# ==============================================================================
#      ** NAYA CODE ** - YEH CSS FILE KO TEMPLATES FOLDER SE LOAD KAREGA
# ==============================================================================
@app.route('/<path:filename>')
def serve_static_files_from_templates(filename):
    # Yeh Flask ko batata hai ke 'style.css' jaisi files ko 'templates' folder se bhejo.
    return send_from_directory('templates', filename)

# --- This starts the web server ---
if __name__ == '__main__':
    app.run(debug=True)