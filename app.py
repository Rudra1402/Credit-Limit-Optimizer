from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model and preprocessor
with open('credit_limit_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('preprocessor.pkl', 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON data
        data = {
            'Customer_Age': int(request.form['Customer_Age']),
            'Gender': request.form['Gender'],
            'Dependent_count': int(request.form['Dependent_count']),
            'Education_Level': request.form['Education_Level'],
            'Marital_Status': request.form['Marital_Status'],
            'Income_Category': request.form['Income_Category'],
            'Card_Category': request.form['Card_Category'],
            'Months_on_book': int(request.form['Months_on_book']),
            'Total_Relationship_Count': int(request.form['Total_Relationship_Count']),
            'Months_Inactive_12_mon': int(request.form['Months_Inactive_12_mon']),
            'Contacts_Count_12_mon': int(request.form['Contacts_Count_12_mon']),
            'Credit_Limit': float(request.form['Credit_Limit']),
            'Total_Revolving_Bal': int(request.form['Total_Revolving_Bal']),
            'Avg_Open_To_Buy': float(request.form['Avg_Open_To_Buy']),
            'Total_Amt_Chng_Q4_Q1': float(request.form['Total_Amt_Chng_Q4_Q1']),
            'Total_Trans_Amt': int(request.form['Total_Trans_Amt']),
            'Total_Trans_Ct': int(request.form['Total_Trans_Ct']),
            'Total_Ct_Chng_Q4_Q1': float(request.form['Total_Ct_Chng_Q4_Q1']),
            'Avg_Utilization_Ratio': float(request.form['Avg_Utilization_Ratio'])
        }
        
        # Convert input data to DataFrame
        input_data = pd.DataFrame(data, index=[0])
        
        # Preprocess the input data
        input_data_preprocessed = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_data_preprocessed)
        
        # Return the prediction as JSON
        return render_template('index.html', prediction=f'Predicted Credit Limit: {prediction[0]:.2f}')
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
