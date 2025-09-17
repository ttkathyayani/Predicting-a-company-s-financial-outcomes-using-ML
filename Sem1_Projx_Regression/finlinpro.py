#import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# Load the dataset
dataset = pd.read_csv('financial_data.csv')

# View the structure of the dataset
print(dataset.head(1))

'''dataframe.head(n)
Return Value
A DataFrame with headers and the specified number of rows.
'''

# Visualize the relationships
plt.scatter(dataset['Revenue (in $)'], dataset['Profit (in $)'], color='blue', label='Revenue vs Profit')
plt.scatter(dataset['Expenses (in $)'], dataset['Profit (in $)'], color='red', label='Expenses vs Profit')
plt.title('Financial Relationships')
plt.xlabel('Financial Metrics')
plt.ylabel('Profit (in $)')
plt.legend()


# Save the plot as a PDF or PNG file
plt.savefig("plotoutput.pdf")  # This saves it as a PDF file
plt.savefig("plotoutput.png")  # Optionally, save as a PNG file

# Show the plot
plt.show()

'''
plt.savefig(sys.stdout.buffer) saves the plot as binary data and writes it to the standard output.
sys.stdout.flush() ensures that all buffered data is written immediately, making sure the image is fully output.
gives garbage
'''

# Define independent and dependent variables
X = dataset[['Revenue (in $)', 'Expenses (in $)']].values
y = dataset['Profit (in $)'].values

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Features Trained80: {X_train}")
print(f"Features Test20: {X_test}")
print(f"target Trained80: {y_train}")
print(f"target Test20: {y_test}")


# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Output the intercept and coefficients
print(f"Intercept: {regressor.intercept_}")
print(f"Coefficients: {regressor.coef_}")

# Make predictions on the test data
y_pred = regressor.predict(X_test)

print(f"feature test and target prediction: {y_pred}")


# Compare actual vs predicted
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)


# Evaluate the model
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) #This requires imprting numpy
