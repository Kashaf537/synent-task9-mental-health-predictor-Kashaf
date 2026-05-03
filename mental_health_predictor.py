"""
Mental Health Predictor - Complete Flask Application
Single file - Just run this!
"""

# ============================================
# IMPORTS
# ============================================

from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

# ============================================
# HTML TEMPLATE (Embedded - FIXED)
# ============================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mental Health Predictor | Lifestyle Analysis</title>

    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container { max-width: 1200px; margin: 0 auto; }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }

        .header h1 { font-size: 2.5rem; margin-bottom: 10px; }
        .header p { font-size: 1.1rem; opacity: 0.9; }

        .main-card {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }

        .form-section { padding: 30px; }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }

        .input-group { margin-bottom: 15px; }

        .input-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #333;
        }

        .input-group input, .input-group select {
            width: 100%;
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
        }

        .btn-predict {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border: none;
            border-radius: 50px;
            font-size: 18px;
            cursor: pointer;
            width: 100%;
            margin-top: 20px;
        }

        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }

        .results-section {
            background: #f8f9fa;
            padding: 30px;
            border-top: 3px solid #667eea;
        }

        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }

        .result-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }

        .risk-low { color: green; }
        .risk-high { color: red; }
        .risk-medium { color: orange; }

        .overall-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin-top: 20px;
            text-align: center;
        }

        .recommendations {
            margin-top: 20px;
            padding: 15px;
            background: #e8f5e9;
            border-radius: 10px;
        }

        .recommendations ul {
            margin-left: 20px;
            margin-top: 10px;
        }

        .recommendations li {
            margin: 5px 0;
        }

        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 4px solid #f44336;
        }

        h2 {
            margin-bottom: 20px;
            color: #333;
        }

        h3 {
            margin-bottom: 10px;
            color: #555;
        }

        h4 {
            color: #2e7d32;
        }

        .probability {
            font-size: 1.2rem;
            font-weight: bold;
            margin-top: 10px;
        }
    </style>
</head>

<body>
<div class="container">

    <div class="header">
        <h1>🧠 Mental Health Risk Predictor</h1>
        <p>Based on your lifestyle habits | Powered by Machine Learning</p>
    </div>

    <div class="main-card">

        <div class="form-section">
            <h2>Tell us about your lifestyle</h2>
            <p>Fill in the details below for personalized risk assessment</p>
            
            {% if error %}
                <div class="error">⚠️ {{ error }}</div>
            {% endif %}

            <form method="POST" action="/predict">
                <div class="form-grid">
                    <div class="input-group">
                        <label>Age</label>
                        <input type="number" name="age" value="{{ form_data.get('age', '30') }}" required>
                    </div>
                    
                    <div class="input-group">
                        <label>Gender</label>
                        <select name="gender">
                            <option value="Male" {% if form_data.get('gender') == 'Male' %}selected{% endif %}>Male</option>
                            <option value="Female" {% if form_data.get('gender') == 'Female' %}selected{% endif %}>Female</option>
                        </select>
                    </div>
                    
                    <div class="input-group">
                        <label>Daily Screen Time (hours)</label>
                        <input type="number" name="daily_screen_time" value="{{ form_data.get('daily_screen_time', '6') }}" step="0.5" required>
                    </div>
                    
                    <div class="input-group">
                        <label>Social Media Usage (hours/day)</label>
                        <input type="number" name="social_media_usage" value="{{ form_data.get('social_media_usage', '4') }}" step="0.5" required>
                    </div>
                    
                    <div class="input-group">
                        <label>Night Usage (after 10 PM)</label>
                        <select name="night_usage">
                            <option value="0" {% if form_data.get('night_usage') == '0' %}selected{% endif %}>No</option>
                            <option value="1" {% if form_data.get('night_usage') == '1' %}selected{% endif %}>Yes</option>
                        </select>
                    </div>
                    
                    <div class="input-group">
                        <label>Sleep Hours (per night)</label>
                        <input type="number" name="sleep_hours" value="{{ form_data.get('sleep_hours', '7') }}" step="0.5" required>
                    </div>
                    
                    <div class="input-group">
                        <label>Stress Level (1-10)</label>
                        <input type="number" name="stress_level" value="{{ form_data.get('stress_level', '5') }}" min="1" max="10" required>
                    </div>
                    
                    <div class="input-group">
                        <label>Work/Study Hours (per day)</label>
                        <input type="number" name="work_study_hours" value="{{ form_data.get('work_study_hours', '8') }}" step="0.5" required>
                    </div>
                    
                    <div class="input-group">
                        <label>Physical Activity</label>
                        <select name="physical_activity">
                            <option value="0" {% if form_data.get('physical_activity') == '0' %}selected{% endif %}>Low</option>
                            <option value="1" {% if form_data.get('physical_activity') == '1' %}selected{% endif %}>Medium</option>
                            <option value="2" {% if form_data.get('physical_activity') == '2' %}selected{% endif %}>High</option>
                        </select>
                    </div>
                    
                    <div class="input-group">
                        <label>Social Interaction Score (1-10)</label>
                        <input type="number" name="social_interaction_score" value="{{ form_data.get('social_interaction_score', '7') }}" min="1" max="10" required>
                    </div>
                    
                    <div class="input-group">
                        <label>Caffeine Intake (cups/day)</label>
                        <input type="number" name="caffeine_intake" value="{{ form_data.get('caffeine_intake', '2') }}" required>
                    </div>
                    
                    <div class="input-group">
                        <label>Smoking</label>
                        <select name="smoking">
                            <option value="0" {% if form_data.get('smoking') == '0' %}selected{% endif %}>No</option>
                            <option value="1" {% if form_data.get('smoking') == '1' %}selected{% endif %}>Yes</option>
                        </select>
                    </div>
                    
                    <div class="input-group">
                        <label>Alcohol Consumption</label>
                        <select name="alcohol">
                            <option value="0" {% if form_data.get('alcohol') == '0' %}selected{% endif %}>No</option>
                            <option value="1" {% if form_data.get('alcohol') == '1' %}selected{% endif %}>Yes</option>
                        </select>
                    </div>
                </div>

                <button type="submit" class="btn-predict">Predict My Risk</button>
            </form>
        </div>

        <!-- FIXED RESULTS BLOCK - Safe checking for overall_risk -->
        {% if prediction and prediction.overall_risk %}
        <div class="results-section">
            <h2>📊 Your Mental Health Risk Assessment</h2>
            
            <div class="results-grid">
                <div class="result-card">
                    <h3>😔 Depression Risk</h3>
                    <div class="probability">{{ prediction.depression.probability }}%</div>
                    <p>{{ prediction.depression.message }}</p>
                </div>
                
                <div class="result-card">
                    <h3>😰 Anxiety Risk</h3>
                    <div class="probability">{{ prediction.anxiety.probability }}%</div>
                    <p>{{ prediction.anxiety.message }}</p>
                </div>
                
                <div class="result-card">
                    <h3>😫 Burnout Risk</h3>
                    <div class="probability">{{ prediction.burnout.probability }}%</div>
                    <p>{{ prediction.burnout.message }}</p>
                </div>
            </div>
            
            <div class="overall-card">
                <h3>🎯 Overall Mental Health Risk</h3>
                <div style="font-size: 3rem; font-weight: bold;">
                    {{ prediction.overall_risk.score }}%
                </div>
                <p style="font-size: 1.2rem; margin-top: 10px;">
                    Risk Level: 
                    <strong class="risk-{{ prediction.overall_risk.level|lower }}">
                        {{ prediction.overall_risk.level }}
                    </strong>
                </p>
            </div>
            
            <div class="recommendations">
                <h4>💡 Personalized Recommendations</h4>
                <ul>
                    {% for rec in prediction.overall_risk.recommendation %}
                        <li>{{ rec }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>
        {% elif prediction %}
        <div class="error">
            ⚠️ Unable to display results. Please try again.
        </div>
        {% endif %}
        
    </div>
</div>
</body>
</html>
'''

# ============================================
# INITIALIZE FLASK APP
# ============================================

app = Flask(__name__)

depression_model = None
anxiety_model = None
burnout_model = None
scaler = None

FEATURE_COLUMNS = [
    'Age', 'Gender', 'Daily_Screen_Time', 'Social_Media_Usage',
    'Night_Usage', 'Sleep_Hours', 'Stress_Level', 'Work_Study_Hours',
    'Physical_Activity', 'Social_Interaction_Score', 'Caffeine_Intake',
    'Smoking', 'Alcohol'
]

# ============================================
# TRAIN MODELS
# ============================================

def train_models():
    global depression_model, anxiety_model, burnout_model, scaler

    print("\n🤖 Training Machine Learning Models...")
    print("-" * 40)

    # Check if dataset exists
    if not os.path.exists('mental_health_lifestyle_2000.csv'):
        print("❌ Dataset 'mental_health_lifestyle_2000.csv' not found!")
        print("📝 Please ensure the dataset file is in the same directory.")
        return False

    try:
        # Load and prepare data
        df = pd.read_csv('mental_health_lifestyle_2000.csv')
        print(f"✅ Dataset loaded: {len(df)} samples")
        
        df = df.dropna()
        
        # Encode categorical variables
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
        df['Physical_Activity'] = df['Physical_Activity'].map({'Low': 0, 'Medium': 1, 'High': 2})
        
        # Prepare features and targets
        X = df[FEATURE_COLUMNS]
        y1 = df['Depression']
        y2 = df['Anxiety']
        y3 = df['Burnout']
        
        # Split data
        X_train, X_test, y1_tr, y1_te, y2_tr, y2_te, y3_tr, y3_te = train_test_split(
            X, y1, y2, y3, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train models
        print("🔄 Training models...")
        depression_model = RandomForestClassifier(n_estimators=100, random_state=42)
        anxiety_model = RandomForestClassifier(n_estimators=100, random_state=42)
        burnout_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        depression_model.fit(X_train, y1_tr)
        anxiety_model.fit(X_train, y2_tr)
        burnout_model.fit(X_train, y3_tr)
        
        # Calculate accuracies
        dep_acc = depression_model.score(X_test, y1_te) * 100
        anx_acc = anxiety_model.score(X_test, y2_te) * 100
        burn_acc = burnout_model.score(X_test, y3_te) * 100
        
        print(f"✅ Depression Model Accuracy: {dep_acc:.1f}%")
        print(f"✅ Anxiety Model Accuracy: {anx_acc:.1f}%")
        print(f"✅ Burnout Model Accuracy: {burn_acc:.1f}%")
        print("-" * 40)
        print("🎉 All models trained successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error training models: {str(e)}")
        return False

# ============================================
# HOME ROUTE (FIXED)
# ============================================

@app.route('/')
def home():
    return render_template_string(
        HTML_TEMPLATE,
        form_data={},
        prediction=None,
        error=None
    )

# ============================================
# PREDICT ROUTE (FIXED)
# ============================================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Validate required fields
        required_fields = ['age', 'gender', 'daily_screen_time', 'social_media_usage', 
                          'night_usage', 'sleep_hours', 'stress_level', 'work_study_hours',
                          'physical_activity', 'social_interaction_score', 'caffeine_intake',
                          'smoking', 'alcohol']
        
        for field in required_fields:
            if field not in form_data:
                raise ValueError(f"Missing field: {field}")
        
        # Prepare input data
        input_data = {
            'Age': float(form_data['age']),
            'Gender': 0 if form_data['gender'] == 'Male' else 1,
            'Daily_Screen_Time': float(form_data['daily_screen_time']),
            'Social_Media_Usage': float(form_data['social_media_usage']),
            'Night_Usage': int(form_data['night_usage']),
            'Sleep_Hours': float(form_data['sleep_hours']),
            'Stress_Level': int(form_data['stress_level']),
            'Work_Study_Hours': float(form_data['work_study_hours']),
            'Physical_Activity': int(form_data['physical_activity']),
            'Social_Interaction_Score': int(form_data['social_interaction_score']),
            'Caffeine_Intake': int(form_data['caffeine_intake']),
            'Smoking': int(form_data['smoking']),
            'Alcohol': int(form_data['alcohol'])
        }
        
        # Scale input
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        
        # Get predictions and probabilities
        dep_pred = depression_model.predict(input_scaled)[0]
        anx_pred = anxiety_model.predict(input_scaled)[0]
        burn_pred = burnout_model.predict(input_scaled)[0]
        
        dep_prob = depression_model.predict_proba(input_scaled)[0][1] * 100
        anx_prob = anxiety_model.predict_proba(input_scaled)[0][1] * 100
        burn_prob = burnout_model.predict_proba(input_scaled)[0][1] * 100
        
        # Create result dictionary
        result = {
            'depression': {
                'prediction': int(dep_pred),
                'probability': round(dep_prob, 2),
                'message': '⚠️ High risk of Depression' if dep_pred else '✅ Low risk of Depression'
            },
            'anxiety': {
                'prediction': int(anx_pred),
                'probability': round(anx_prob, 2),
                'message': '⚠️ High risk of Anxiety' if anx_pred else '✅ Low risk of Anxiety'
            },
            'burnout': {
                'prediction': int(burn_pred),
                'probability': round(burn_prob, 2),
                'message': '⚠️ High risk of Burnout' if burn_pred else '✅ Low risk of Burnout'
            }
        }
        
        # Calculate overall risk score
        overall_score = (dep_prob + anx_prob + burn_prob) / 3
        
        # Determine risk level and recommendations
        if overall_score > 60:
            level = "High"
            recommendations = [
                "Consider consulting a mental health professional",
                "Practice stress management techniques like meditation",
                "Establish a regular sleep schedule (7-9 hours)",
                "Reduce screen time and social media usage",
                "Increase physical activity and social interactions"
            ]
        elif overall_score > 30:
            level = "Medium"
            recommendations = [
                "Monitor your symptoms and practice self-care",
                "Take regular breaks from work/studies",
                "Maintain a balanced diet and exercise routine",
                "Connect with friends and family regularly",
                "Limit caffeine intake after 4 PM"
            ]
        else:
            level = "Low"
            recommendations = [
                "Continue maintaining healthy lifestyle habits",
                "Exercise regularly (30 mins/day)",
                "Practice good sleep hygiene",
                "Stay socially connected",
                "Take breaks when feeling overwhelmed"
            ]
        
        # Add overall risk to result
        result['overall_risk'] = {
            'score': round(overall_score, 2),
            'level': level,
            'recommendation': recommendations
        }
        
        # Render template with results
        return render_template_string(
            HTML_TEMPLATE,
            prediction=result,
            form_data=form_data,
            error=None
        )
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return render_template_string(
            HTML_TEMPLATE,
            prediction=None,
            form_data=request.form.to_dict() if request.form else {},
            error=f"Error making prediction: {str(e)}"
        )

# ============================================
# HEALTH CHECK ROUTE
# ============================================

@app.route('/health')
def health():
    return jsonify({
        'status': 'running',
        'models_loaded': all([depression_model, anxiety_model, burnout_model, scaler])
    })

# ============================================
# MAIN (FIXED)
# ============================================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🧠 MENTAL HEALTH RISK PREDICTOR")
    print("="*50)
    
    # Train models
    if train_models():
        print("\n🚀 Starting Flask server...")
        print("📍 Server running at: http://127.0.0.1:5000")
        print("📍 Local URL: http://localhost:5000")
        print("\n⚠️  Make sure your dataset 'mental_health_lifestyle_2000.csv' is in the same directory")
        print("="*50 + "\n")
        
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5000,
            use_reloader=False
        )
    else:
        print("\n❌ Failed to start application. Please check the dataset file.")
        print("📝 Required: 'mental_health_lifestyle_2000.csv' in the current directory")