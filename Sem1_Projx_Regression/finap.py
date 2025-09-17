from flask import Flask, request, jsonify
import numpy as np
import pickle  # To save/load your model

app = Flask(__name__)

# Load trained model
model = pickle.load(open('financial_model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get data from the request
    print(data)  # Debug: Print the incoming data
    revenue = float(data['Revenue'])
    expenses = float(data['Expenses'])
    
    # Make prediction
    prediction = model.predict(np.array([[revenue, expenses]]))
    return jsonify({'Profit Prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)