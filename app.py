from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved model
with open(r'F:\ML Ops\stroke_prediction\xgb_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the saved scaler
with open(r'F:\ML Ops\stroke_prediction\scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load the saved columns
with open(r'F:\ML Ops\stroke_prediction\columns.pkl', 'rb') as file:
    columns = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = {
        'age': float(request.form['age']),
        'hypertension': int(request.form['hypertension']),
        'heart_disease': int(request.form['heart_disease']),
        'avg_glucose_level': float(request.form['avg_glucose_level']),
        'bmi': float(request.form['bmi']),
        'gender_Male': int(request.form['gender'] == 'Male'),
        'gender_Female': int(request.form['gender'] == 'Female'),
        'gender_Other': int(request.form['gender'] == 'Other'),
        'ever_married_Yes': int(request.form['ever_married'] == 'Yes'),
        'ever_married_No': int(request.form['ever_married'] == 'No'),
        'work_type_Never_worked': int(request.form['work_type'] == 'Never_worked'),
        'work_type_Private': int(request.form['work_type'] == 'Private'),
        'work_type_Self-employed': int(request.form['work_type'] == 'Self-employed'),
        'work_type_children': int(request.form['work_type'] == 'children'),
        'Residence_type_Urban': int(request.form['Residence_type'] == 'Urban'),
        'Residence_type_Rural': int(request.form['Residence_type'] == 'Rural'),
        'smoking_status_formerly smoked': int(request.form['smoking_status'] == 'formerly smoked'),
        'smoking_status_never smoked': int(request.form['smoking_status'] == 'never smoked'),
        'smoking_status_smokes': int(request.form['smoking_status'] == 'smokes')
        }


        df = pd.DataFrame(data, index=[0])

        # Standardize numerical variables
        numerical_variables = ['age', 'avg_glucose_level', 'bmi']
        df[numerical_variables] = scaler.transform(df[numerical_variables])

        # Ensure all columns are present in the DataFrame
        df = pd.get_dummies(df)
        df = df.reindex(columns=columns, fill_value=0)

        # Predict the result
        prediction = model.predict(df)
        result = 'Stroke' if prediction[0] == 1 else 'No Stroke'
    except Exception as e:
        result = str(e)

    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == '__main__':
    app.run(debug=True)
