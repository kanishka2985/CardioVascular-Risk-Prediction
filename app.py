from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for session

# Load the pre-trained model
try:
    with open('care4heart.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    raise Exception("Model file not found. Please ensure 'care4heart.pkl' exists in the application directory.")
except Exception as e:
    raise Exception(f"Error loading model: {str(e)}")

# Try to get feature names
if hasattr(model, 'feature_names_in_'):
    feature_names = list(model.feature_names_in_)
elif hasattr(model, 'feature_names'):
    feature_names = list(model.feature_names)
    if feature_names is None:
        raise Exception("Model's feature_names attribute is None. Please specify feature names manually.")
else:
    # Manually specify your feature names here, in the order used for training
    feature_names = [
        'age', 'education', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol',
        'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 'pulse_pressure', 'is_obese', 'high_cholesterol',
        'sex_female', 'sex_male', 'is_smoking_no', 'is_smoking_yes',
        'glucose_status_Diabetes', 'glucose_status_Normal', 'glucose_status_Pre-diabetes',
        'smoking_intensity_Heavy Smoker', 'smoking_intensity_Light Smoker',
        'smoking_intensity_Moderate Smoker', 'smoking_intensity_Non-Smoker'
    ]

print("Model expects these features:")
print(feature_names)

@app.route('/')
def home():
    result_data = session.pop('result_data', None)
    if result_data:
        # Determine risk_level for tips and doctors
        risk_level = 'low'
        if 'result_type' in result_data:
            if result_data['result_type'] == 'at-risk':
                risk_level = 'high'
            elif result_data['result_type'] == 'moderate-risk':
                risk_level = 'moderate'
        # Inline tips_data and doctors_data (copied from health_tips and doctor_recommendations)
        tips_data = {
            'low': {
                'title': 'Maintain Your Healthy Lifestyle',
                'tips': [
                    {'icon': '🏃‍♂', 'title': 'Regular Exercise', 'description': 'Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity per week.'},
                    {'icon': '🥗', 'title': 'Heart-Healthy Diet', 'description': 'Focus on fruits, vegetables, whole grains, lean proteins, and healthy fats. Limit sodium, saturated fats, and added sugars.'},
                    {'icon': '😴', 'title': 'Quality Sleep', 'description': 'Get 7-9 hours of quality sleep per night to support heart health and overall well-being.'},
                    {'icon': '🧘‍♀', 'title': 'Stress Management', 'description': 'Practice stress-reduction techniques like meditation, deep breathing, or yoga.'},
                    {'icon': '🚭', 'title': 'Avoid Smoking', 'description': 'If you don\'t smoke, don\'t start. If you do smoke, seek help to quit.'},
                    {'icon': '⚖', 'title': 'Maintain Healthy Weight', 'description': 'Keep your BMI in the healthy range (18.5-24.9) through diet and exercise.'}
                ]
            },
            'moderate': {
                'title': 'Lifestyle Changes for Better Health',
                'tips': [
                    {'icon': '🏥', 'title': 'Regular Check-ups', 'description': 'Schedule regular visits with your healthcare provider to monitor your health status.'},
                    {'icon': '📊', 'title': 'Monitor Key Metrics', 'description': 'Track your blood pressure, cholesterol, and blood sugar levels regularly.'},
                    {'icon': '🚶‍♂', 'title': 'Increase Physical Activity', 'description': 'Start with walking 30 minutes daily and gradually increase intensity.'},
                    {'icon': '🥬', 'title': 'Diet Modifications', 'description': 'Reduce salt intake, increase fiber consumption, and limit processed foods.'},
                    {'icon': '💊', 'title': 'Medication Adherence', 'description': 'If prescribed medications, take them exactly as directed by your doctor.'},
                    {'icon': '📱', 'title': 'Health Apps', 'description': 'Use health tracking apps to monitor your progress and stay motivated.'}
                ]
            },
            'high': {
                'title': 'Immediate Action Required',
                'tips': [
                    {'icon': '🚨', 'title': 'Emergency Contact', 'description': 'If you experience chest pain, shortness of breath, or other concerning symptoms, call emergency services immediately.'},
                    {'icon': '👨‍⚕', 'title': 'Doctor Consultation', 'description': 'Schedule an appointment with a cardiologist or primary care physician as soon as possible.'},
                    {'icon': '💊', 'title': 'Medication Review', 'description': 'Discuss with your doctor about medications that may help manage your risk factors.'},
                    {'icon': '🏃‍♂', 'title': 'Supervised Exercise', 'description': 'Consider cardiac rehabilitation or supervised exercise programs.'},
                    {'icon': '📋', 'title': 'Health Monitoring', 'description': 'Keep detailed records of your symptoms, medications, and vital signs.'},
                    {'icon': '👨‍👩‍👧‍👦', 'title': 'Family Support', 'description': 'Involve family members in your health journey for support and accountability.'}
                ]
            }
        }
        doctors_data = {
            'high': {
                'title': 'Recommended Healthcare Providers',
                'subtitle': 'Based on your risk assessment, we recommend consulting with these specialists:',
                'doctors': [
                    {'name': 'Dr. Sarah Johnson', 'specialty': 'Cardiologist', 'experience': '15+ years', 'location': 'Heart & Vascular Institute', 'phone': '(555) 123-4567', 'rating': '4.9/5', 'availability': 'Next available: Tomorrow'},
                    {'name': 'Dr. Michael Chen', 'specialty': 'Cardiovascular Surgeon', 'experience': '20+ years', 'location': 'Cardiac Care Center', 'phone': '(555) 234-5678', 'rating': '4.8/5', 'availability': 'Next available: This week'},
                    {'name': 'Dr. Emily Rodriguez', 'specialty': 'Preventive Cardiologist', 'experience': '12+ years', 'location': 'Preventive Medicine Clinic', 'phone': '(555) 345-6789', 'rating': '4.7/5', 'availability': 'Next available: Next week'},
                    {'name': 'Dr. James Wilson', 'specialty': 'Interventional Cardiologist', 'experience': '18+ years', 'location': 'Advanced Cardiac Institute', 'phone': '(555) 456-7890', 'rating': '4.9/5', 'availability': 'Next available: This week'}
                ]
            },
            'moderate': {
                'title': 'Recommended Healthcare Providers',
                'subtitle': 'Consider consulting with these healthcare professionals:',
                'doctors': [
                    {'name': 'Dr. Lisa Thompson', 'specialty': 'Primary Care Physician', 'experience': '10+ years', 'location': 'Family Health Center', 'phone': '(555) 567-8901', 'rating': '4.6/5', 'availability': 'Next available: This week'},
                    {'name': 'Dr. Robert Davis', 'specialty': 'Cardiologist', 'experience': '12+ years', 'location': 'Heart Health Clinic', 'phone': '(555) 678-9012', 'rating': '4.7/5', 'availability': 'Next available: Next week'},
                    {'name': 'Dr. Amanda Foster', 'specialty': 'Preventive Medicine', 'experience': '8+ years', 'location': 'Wellness Center', 'phone': '(555) 789-0123', 'rating': '4.5/5', 'availability': 'Next available: This week'}
                ]
            },
            'low': {
                'title': 'Preventive Care Providers',
                'subtitle': 'For ongoing preventive care, consider these healthcare professionals:',
                'doctors': [
                    {'name': 'Dr. Jennifer Lee', 'specialty': 'Primary Care Physician', 'experience': '8+ years', 'location': 'Community Health Center', 'phone': '(555) 890-1234', 'rating': '4.6/5', 'availability': 'Next available: This week'},
                    {'name': 'Dr. David Brown', 'specialty': 'Family Medicine', 'experience': '12+ years', 'location': 'Family Practice Associates', 'phone': '(555) 901-2345', 'rating': '4.7/5', 'availability': 'Next available: Next week'},
                    {'name': 'Dr. Maria Garcia', 'specialty': 'Preventive Medicine', 'experience': '6+ years', 'location': 'Wellness & Prevention Clinic', 'phone': '(555) 012-3456', 'rating': '4.5/5', 'availability': 'Next available: This week'}
                ]
            }
        }
        result_data['tips_data'] = tips_data[risk_level]
        result_data['doctors_data'] = doctors_data[risk_level]
        return render_template('index.html', **result_data)
    return render_template('index.html')

@app.route('/health-tips')
def health_tips():
    """Return health tips based on risk level"""
    risk_level = request.args.get('risk_level', 'low')
    
    tips_data = {
        'low': {
            'title': 'Maintain Your Healthy Lifestyle',
            'tips': [
                {
                    'icon': '🏃‍♂',
                    'title': 'Regular Exercise',
                    'description': 'Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity per week.'
                },
                {
                    'icon': '🥗',
                    'title': 'Heart-Healthy Diet',
                    'description': 'Focus on fruits, vegetables, whole grains, lean proteins, and healthy fats. Limit sodium, saturated fats, and added sugars.'
                },
                {
                    'icon': '😴',
                    'title': 'Quality Sleep',
                    'description': 'Get 7-9 hours of quality sleep per night to support heart health and overall well-being.'
                },
                {
                    'icon': '🧘‍♀',
                    'title': 'Stress Management',
                    'description': 'Practice stress-reduction techniques like meditation, deep breathing, or yoga.'
                },
                {
                    'icon': '🚭',
                    'title': 'Avoid Smoking',
                    'description': 'If you don\'t smoke, don\'t start. If you do smoke, seek help to quit.'
                },
                {
                    'icon': '⚖',
                    'title': 'Maintain Healthy Weight',
                    'description': 'Keep your BMI in the healthy range (18.5-24.9) through diet and exercise.'
                }
            ]
        },
        'moderate': {
            'title': 'Lifestyle Changes for Better Health',
            'tips': [
                {
                    'icon': '🏥',
                    'title': 'Regular Check-ups',
                    'description': 'Schedule regular visits with your healthcare provider to monitor your health status.'
                },
                {
                    'icon': '📊',
                    'title': 'Monitor Key Metrics',
                    'description': 'Track your blood pressure, cholesterol, and blood sugar levels regularly.'
                },
                {
                    'icon': '🚶‍♂',
                    'title': 'Increase Physical Activity',
                    'description': 'Start with walking 30 minutes daily and gradually increase intensity.'
                },
                {
                    'icon': '🥬',
                    'title': 'Diet Modifications',
                    'description': 'Reduce salt intake, increase fiber consumption, and limit processed foods.'
                },
                {
                    'icon': '💊',
                    'title': 'Medication Adherence',
                    'description': 'If prescribed medications, take them exactly as directed by your doctor.'
                },
                {
                    'icon': '📱',
                    'title': 'Health Apps',
                    'description': 'Use health tracking apps to monitor your progress and stay motivated.'
                }
            ]
        },
        'high': {
            'title': 'Immediate Action Required',
            'tips': [
                {
                    'icon': '🚨',
                    'title': 'Emergency Contact',
                    'description': 'If you experience chest pain, shortness of breath, or other concerning symptoms, call emergency services immediately.'
                },
                {
                    'icon': '👨‍⚕',
                    'title': 'Doctor Consultation',
                    'description': 'Schedule an appointment with a cardiologist or primary care physician as soon as possible.'
                },
                {
                    'icon': '💊',
                    'title': 'Medication Review',
                    'description': 'Discuss with your doctor about medications that may help manage your risk factors.'
                },
                {
                    'icon': '🏃‍♂',
                    'title': 'Supervised Exercise',
                    'description': 'Consider cardiac rehabilitation or supervised exercise programs.'
                },
                {
                    'icon': '📋',
                    'title': 'Health Monitoring',
                    'description': 'Keep detailed records of your symptoms, medications, and vital signs.'
                },
                {
                    'icon': '👨‍👩‍👧‍👦',
                    'title': 'Family Support',
                    'description': 'Involve family members in your health journey for support and accountability.'
                }
            ]
        }
    }
    
    return jsonify(tips_data.get(risk_level, tips_data['low']))

@app.route('/doctor-recommendations')
def doctor_recommendations():
    """Return doctor recommendations based on risk level"""
    risk_level = request.args.get('risk_level', 'high')
    
    doctors_data = {
        'high': {
            'title': 'Recommended Healthcare Providers',
            'subtitle': 'Based on your risk assessment, we recommend consulting with these specialists:',
            'doctors': [
                {
                    'name': 'Dr. Sarah Johnson',
                    'specialty': 'Cardiologist',
                    'experience': '15+ years',
                    'location': 'Heart & Vascular Institute',
                    'phone': '(555) 123-4567',
                    'rating': '4.9/5',
                    'availability': 'Next available: Tomorrow'
                },
                {
                    'name': 'Dr. Michael Chen',
                    'specialty': 'Cardiovascular Surgeon',
                    'experience': '20+ years',
                    'location': 'Cardiac Care Center',
                    'phone': '(555) 234-5678',
                    'rating': '4.8/5',
                    'availability': 'Next available: This week'
                },
                {
                    'name': 'Dr. Emily Rodriguez',
                    'specialty': 'Preventive Cardiologist',
                    'experience': '12+ years',
                    'location': 'Preventive Medicine Clinic',
                    'phone': '(555) 345-6789',
                    'rating': '4.7/5',
                    'availability': 'Next available: Next week'
                },
                {
                    'name': 'Dr. James Wilson',
                    'specialty': 'Interventional Cardiologist',
                    'experience': '18+ years',
                    'location': 'Advanced Cardiac Institute',
                    'phone': '(555) 456-7890',
                    'rating': '4.9/5',
                    'availability': 'Next available: This week'
                }
            ]
        },
        'moderate': {
            'title': 'Recommended Healthcare Providers',
            'subtitle': 'Consider consulting with these healthcare professionals:',
            'doctors': [
                {
                    'name': 'Dr. Lisa Thompson',
                    'specialty': 'Primary Care Physician',
                    'experience': '10+ years',
                    'location': 'Family Health Center',
                    'phone': '(555) 567-8901',
                    'rating': '4.6/5',
                    'availability': 'Next available: This week'
                },
                {
                    'name': 'Dr. Robert Davis',
                    'specialty': 'Cardiologist',
                    'experience': '12+ years',
                    'location': 'Heart Health Clinic',
                    'phone': '(555) 678-9012',
                    'rating': '4.7/5',
                    'availability': 'Next available: Next week'
                },
                {
                    'name': 'Dr. Amanda Foster',
                    'specialty': 'Preventive Medicine',
                    'experience': '8+ years',
                    'location': 'Wellness Center',
                    'phone': '(555) 789-0123',
                    'rating': '4.5/5',
                    'availability': 'Next available: This week'
                }
            ]
        },
        'low': {
            'title': 'Preventive Care Providers',
            'subtitle': 'For ongoing preventive care, consider these healthcare professionals:',
            'doctors': [
                {
                    'name': 'Dr. Jennifer Lee',
                    'specialty': 'Primary Care Physician',
                    'experience': '8+ years',
                    'location': 'Community Health Center',
                    'phone': '(555) 890-1234',
                    'rating': '4.6/5',
                    'availability': 'Next available: This week'
                },
                {
                    'name': 'Dr. David Brown',
                    'specialty': 'Family Medicine',
                    'experience': '12+ years',
                    'location': 'Family Practice Associates',
                    'phone': '(555) 901-2345',
                    'rating': '4.7/5',
                    'availability': 'Next available: Next week'
                },
                {
                    'name': 'Dr. Maria Garcia',
                    'specialty': 'Preventive Medicine',
                    'experience': '6+ years',
                    'location': 'Wellness & Prevention Clinic',
                    'phone': '(555) 012-3456',
                    'rating': '4.5/5',
                    'availability': 'Next available: This week'
                }
            ]
        }
    }
    
    return jsonify(doctors_data.get(risk_level, doctors_data['low']))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Numeric features
        age = float(request.form.get('age', 0))
        education = float(request.form.get('education', 0))
        cigsPerDay = float(request.form.get('cigsPerDay', 0))
        totChol = float(request.form.get('totChol', 0))
        sysBP = float(request.form.get('sysBP', 0))
        diaBP = float(request.form.get('diaBP', 0))
        BMI = float(request.form.get('BMI', 0))
        heartRate = float(request.form.get('heartRate', 0))
        glucose = float(request.form.get('glucose', 0))
        pulse_pressure = sysBP - diaBP

        # Binary features
        BPMeds = 1 if request.form.get('BPMeds', 'no') == 'yes' else 0
        prevalentStroke = 1 if request.form.get('prevalentStroke', 'no') == 'yes' else 0
        prevalentHyp = 1 if request.form.get('prevalentHyp', 'no') == 'yes' else 0
        diabetes = 1 if request.form.get('diabetes', 'no') == 'yes' else 0
        is_obese = 1 if request.form.get('is_obese', 'no') == 'yes' else 0
        high_cholesterol = 1 if request.form.get('high_cholesterol', 'no') == 'yes' else 0

        # One-hot encoding for sex
        sex = request.form.get('sex', 'male')
        sex_female = 1 if sex == 'female' else 0
        sex_male = 1 if sex == 'male' else 0

        # One-hot encoding for is_smoking
        is_smoking = request.form.get('is_smoking', 'no')
        is_smoking_no = 1 if is_smoking == 'no' else 0
        is_smoking_yes = 1 if is_smoking == 'yes' else 0

        # One-hot encoding for glucose_status
        glucose_status = request.form.get('glucose_status', 'Normal')
        glucose_status_Diabetes = 1 if glucose_status == 'Diabetes' else 0
        glucose_status_Normal = 1 if glucose_status == 'Normal' else 0
        glucose_status_Pre_diabetes = 1 if glucose_status == 'Pre-diabetes' else 0

        # One-hot encoding for smoking_intensity
        smoking_intensity = request.form.get('smoking_intensity', 'Non-Smoker')
        smoking_intensity_Heavy = 1 if smoking_intensity == 'Heavy Smoker' else 0
        smoking_intensity_Light = 1 if smoking_intensity == 'Light Smoker' else 0
        smoking_intensity_Moderate = 1 if smoking_intensity == 'Moderate Smoker' else 0
        smoking_intensity_Non = 1 if smoking_intensity == 'Non-Smoker' else 0

        # Build feature vector in the correct order
        features = [
            age, education, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol,
            sysBP, diaBP, BMI, heartRate, glucose, pulse_pressure, is_obese, high_cholesterol,
            sex_female, sex_male, is_smoking_no, is_smoking_yes,
            glucose_status_Diabetes, glucose_status_Normal, glucose_status_Pre_diabetes,
            smoking_intensity_Heavy, smoking_intensity_Light,
            smoking_intensity_Moderate, smoking_intensity_Non
        ]

        X = pd.DataFrame([features], columns=feature_names)
        print("Features sent to model:")
        print(X)

        # Predict
        pred = model.predict(X)[0]
        risk_percent = int(model.predict_proba(X)[0][1] * 100) if hasattr(model, "predict_proba") else None

        # Interpret prediction
        if pred == 0:
            risk = "Low Risk"
            result_type = "not-at-risk"
            message = "Great job! You are at LOW risk. Maintain a healthy lifestyle to keep your heart strong."
        else:
            risk = "High Risk"
            result_type = "at-risk"
            message = "Based on your profile, you are at HIGH risk for cardiovascular disease. Please consult a cardiologist immediately."
        
        session['result_data'] = {
            'prediction_text': f'Cardiovascular Disease Risk: {risk}',
            'result_message': message,
            'result_type': result_type,
            'risk_percent': risk_percent if risk_percent is not None else ""
        }
        return redirect(url_for('home'))
    except Exception as e:
        error_message = f"Error occurred: {str(e)}"
        session['result_data'] = {'error_message': error_message}
        return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)