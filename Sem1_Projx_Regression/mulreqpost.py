import pandas as pd
import requests

# URL of your Flask app
url = 'http://127.0.0.1:5000/predict'

# Read the CSV file
csv_file = 'financial_requests.csv'
data = pd.read_csv(csv_file)

# Create an empty list to store the results
results = []

# Loop through the data and send POST requests
for index, row in data.iterrows():
    # Check if 'Revenue' and 'Expenses' columns exist
    if 'Revenue' in row and 'Expenses' in row:
        revenue = int(row['Revenue'])
        expenses = int(row['Expenses'])
        
        # Create the payload
        payload = {
            'Revenue': revenue,
            'Expenses': expenses
        }
        
        # Send POST request to Flask server
        response = requests.post(url, json=payload)
        
        # Extract the predicted profit from the response
        predicted_profit = response.json().get('Profit Prediction', 'N/A')
        
        # Append the result (Revenue, Expenses, Predicted Profit) to the list
        results.append({
            'Revenue': revenue,
            'Expenses': expenses,
            'Predicted Profit': predicted_profit
        })
        print(f"Response for row {index}: {response.json()}")
    else:
        print(f"Missing 'Revenue' or 'Expenses' in row {index}")

# Convert the results to a DataFrame
results_df = pd.DataFrame(results)

# Write the results to a CSV file
results_df.to_csv('profit_prediction.csv', index=False)

print("Predictions saved to 'profit_prediction.csv'.")

