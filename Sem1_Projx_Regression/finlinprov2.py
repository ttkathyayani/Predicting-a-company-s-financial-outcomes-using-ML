#import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import PolynomialFeatures  
from sklearn import metrics
import pickle  # For saving the model

# Load the dataset
dataset = pd.read_csv('financial_data.csv')

# View the structure of the dataset
print(dataset.head(6))
#TS: one revenue value is same as another (700000)..Also one revenue and expense are same for a given profit... so revenue points are only 4 versus 6

'''dataframe.head(n)
Return Value
A DataFrame with headers and the specified number of rows.
'''
# Version 2 addons start: Handling missing data

# Version 2 addons start: Handling missing data
dsfilled = dataset.ffill()  # Forward fill missing values
print(f"Data Forward filled: \n{dsfilled}")


"""
dsfilled = dataset.fillna(method='ffill', inplace=False)  
# Fill missing values by forward fill. TS: inplace=False returns a new DataFrame
print(f"Data Forward filled: \n{dsfilled}")

the above in orange is deprecated
# Proceed with the filled dataset for further operations

"""
dataset = dsfilled  # Assign the filled dataset back to the main 'dataset' variable

#version 2 addons ends

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
#VERSION 3 

# Save the trained model
with open('financial_model.pkl', 'wb') as file:
    pickle.dump(regressor, file)

print('Model saved as financial_model.pkl')

#VERSION 3

#ADDED IN VERSION 2
# Incorporate Cross-Validation Here
# ----------------------------------
# Perform 5-fold cross-validation
cross_val_scores = cross_val_score(regressor, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f'Cross-Validation Scores: {cross_val_scores}')
print(f'Cross-Validation MAE: {cross_val_scores.mean()}')


# END ASS VER2


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

#version 2

# Import required libraries for Ridge and Lasso


# Step 2: Define a function to automatically select the best model (Linear, Ridge, or Lasso)
def select_best_model(X_train, y_train, X_test, y_test):
    # Define models with regularization
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5),
        'Lasso': LassoCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5, max_iter=10000)
    }
    
    best_model = None
    best_score = float('inf')
    
    # Iterate through models to evaluate MAE using cross-validation
    for name, model in models.items():
        model.fit(X_train, y_train)  # Train the model
        y_pred = model.predict(X_test)  # Predict on test data
        mae = metrics.mean_absolute_error(y_test, y_pred)  # Calculate MAE
        
        print(f'{name} MAE: {mae}')  # Print MAE for each model
        
        # Compare and store the best model based on MAE
        if mae < best_score:
            best_score = mae
            best_model = (name, model)
    
    # Output the best model and its performance
    print(f"Best Model: {best_model[0]} with MAE: {best_score}")
    return best_model

# Call the function to select the best model
best_model = select_best_model(X_train, y_train, X_test, y_test)

#version 2

# -------------------------------------
# Polynomial Regression Implementation
# -------------------------------------
# Transform to add polynomial features
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# Fit polynomial regression model
regressor_poly = LinearRegression()
regressor_poly.fit(X_poly_train, y_train)

# Make predictions with polynomial regression
y_pred_poly = regressor_poly.predict(X_poly_test)

# Evaluate the polynomial regression model
print(f"Polynomial Regression MAE: {metrics.mean_absolute_error(y_test, y_pred_poly)}")
print(f"Polynomial Regression MSE: {metrics.mean_squared_error(y_test, y_pred_poly)}")
print(f"Polynomial Regression RMSE: {np.sqrt(metrics.mean_squared_error(y_test, y_pred_poly))}")

# Compare actual vs predicted for polynomial regression
comparison_poly = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_poly})
print(comparison_poly)