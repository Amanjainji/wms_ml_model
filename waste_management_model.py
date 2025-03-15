import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from flask import Flask, request, jsonify

# Load Data
df = pd.read_excel("smart_bin_dataset.xlsx", sheet_name='Sheet1')

# Encode categorical columns
label_encoder = LabelEncoder()
df['Bin Status'] = label_encoder.fit_transform(df['Bin Status'])  # Convert 'OK' and 'Needs Collection' to numbers

drop_columns = ['Timestamp', 'RFID Sensor (User ID)', 'Computer Vision Sensor']
df.drop(columns=drop_columns, inplace=True)

# Splitting features and targets
X = df.drop(columns=['Bin Status', 'Percentage of Full (%)', 'Bins Requiring Attention (%)',
                      'Collection Efficiency (%)', 'Total Waste Collected (kg)',
                      'Average Collection Rate per Day', 'Recycling Rate (%)'])

Y_classification = df['Bin Status']
Y_regression = df[['Percentage of Full (%)', 'Bins Requiring Attention (%)', 'Collection Efficiency (%)',
                   'Total Waste Collected (kg)', 'Average Collection Rate per Day', 'Recycling Rate (%)']]

# Split data
X_train, X_test, y_train_class, y_test_class = train_test_split(X, Y_classification, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, Y_regression, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_reg = scaler.fit_transform(X_train_reg)
X_test_reg = scaler.transform(X_test_reg)

# Train models
clf = RandomForestClassifier()
clf.fit(X_train, y_train_class)

reg = RandomForestRegressor()
reg.fit(X_train_reg, y_train_reg)

# Predictions
y_pred_class = clf.predict(X_test)
y_pred_reg = reg.predict(X_test_reg)

# Evaluate models
accuracy = accuracy_score(y_test_class, y_pred_class)
mae = mean_absolute_error(y_test_reg, y_pred_reg)

print(f'Classification Model Accuracy: {accuracy * 100:.2f}%')
print(f'Regression Model MAE: {mae:.2f}')

# Flask API for React Integration
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    scaled_data = scaler.transform([data])
    bin_status_pred = clf.predict(scaled_data)[0]
    reg_preds = reg.predict(scaled_data)[0]
    response = {
        'Bin Status': int(bin_status_pred),
        'Percentage of Full (%)': reg_preds[0],
        'Bins Requiring Attention (%)': reg_preds[1],
        'Collection Efficiency (%)': reg_preds[2],
        'Total Waste Collected (kg)': reg_preds[3],
        'Average Collection Rate per Day': reg_preds[4],
        'Recycling Rate (%)': reg_preds[5]
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
