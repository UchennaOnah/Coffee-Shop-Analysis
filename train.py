# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import tree

# Load the cleaned merged dataset
cleaned_df = joblib.load(r"C:\Users\uchen\PycharmProjects\CoffeeShopAnalysis\cleaned_merged_df.joblib")


# Define features and target variables
# Feature variables
X = cleaned_df[['roast_dark', 'roast_light', 'roast_medium', 'roast_medium_dark',
                'roast_medium_light', 'roast_very_dark',
                'region_africa_arabia', 'region_caribbean', 'region_central_america', 'region_hawaii',
          'region_asia_pacific', 'region_south_america']]

y = cleaned_df['rating']

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'\nModel Performance:')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'R2 Score: {r2:.4f}')

# Determine which attributes are most influential in predicting rating
# Get feature importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.title('Feature importance in Rating Prediction')
plt.xlabel('Importance')
plt.ylabel('Feature')
print(plt.show())

# Save the trained model
joblib.dump(model, 'rating_prediction_model.joblib')
print("Model saved as 'rating_prediction_model.joblib'.")

# To train a decision tree classifier to categorize products
cleaned_df['rating'] = pd.to_numeric(cleaned_df['rating'], errors='coerce')


# Define popularity tiers based on rating
cleaned_df['popularity'] = pd.qcut(cleaned_df['rating'], q=3, labels=['Low', 'Medium', 'High'])
print(cleaned_df['popularity'].value_counts())

# Encode categorical variables
cleaned_df = pd.get_dummies(cleaned_df, columns=['roast', 'location', 'name', 'roaster', 'slug'])

# Define features (X) and target (y) variable
X = cleaned_df.drop(columns=['popularity'])
y = cleaned_df['popularity']

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Training set: {X_train.shape}, Testing set: {X_test.shape}')

# Check the data
print(X_train.head())
print(y_train.head())

# Convert to numeric and handle errors
X_train = X_train.apply(pd.to_numeric, errors='coerce')
y_train = pd.to_numeric(y_train, errors='coerce')

# Fill missing target values with the mode
y_train.fillna(y_train.mode(), inplace=True)



# Train the decision tree classifier
model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'\nModel Accuracy: {accuracy:.4f}')

# Print classification report
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
print(plt.show())

# Plot the decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(model, feature_names=X.columns, class_names=['Low', 'Medium', 'High'], filled=True, rounded=True)
plt.title('Decision Tree Visualization')
print(plt.show())

# Save the decision tree trained model
joblib.dump(model, 'product_popularity_classifier.pkl')
print("Model saved as 'product_popularity_classifier.pkl'.")







