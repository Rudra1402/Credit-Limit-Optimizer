from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model and preprocessor
with open('credit_limit_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('preprocessor.pkl', 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON data
        data = request.json
        
        # Convert input data to DataFrame
        input_data = pd.DataFrame(data, index=[0])
        
        # Preprocess the input data
        input_data_preprocessed = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_data_preprocessed)
        
        # Return the prediction as JSON
        return jsonify({'credit_limit': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
