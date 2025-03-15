import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_absolute_error
from flask import Flask, request, jsonify

# Load dataset
df = pd.read_excel("smart_bin_synthetic.xlsx")

# Encode categorical variable
label_encoder = LabelEncoder()
df['Bin Status'] = label_encoder.fit_transform(df['Bin Status'])  # 0: OK, 1: Needs Collection

# Features & target variables
X = df[['Gas Sensor (ppm)', 'Temperature Sensor (°C)', 'Light Sensor (lux)', 'Ultrasonic Sensor (cm)',
        'Moisture Sensor (%)', 'Weight Sensor (kg)']]
Y_classification = df['Bin Status']
Y_regression = df[['Total Waste Collected (kg)', 'Average Collection Rate per Day (kg)', 'Recycling Rate (%)']]

# Train-test split
X_train, X_test, y_train_class, y_test_class = train_test_split(X, Y_classification, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, Y_regression, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_reg = scaler.fit_transform(X_train_reg)
X_test_reg = scaler.transform(X_test_reg)

# Train classification model
clf = RandomForestClassifier()
clf.fit(X_train, y_train_class)

# Train regression model
reg = RandomForestRegressor()
reg.fit(X_train_reg, y_train_reg)

# Predictions
y_pred_class = clf.predict(X_test)
y_pred_reg = reg.predict(X_test_reg)

# Evaluate models
accuracy = accuracy_score(y_test_class, y_pred_class)
mae = mean_absolute_error(y_test_reg, y_pred_reg)

print(f'Classification Accuracy: {accuracy * 100:.2f}%')
print(f'Regression MAE: {mae:.2f}')

# Clustering for heatmaps
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Latitude', 'Longitude']])

# Generate Heatmap
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Longitude'], y=df['Latitude'], hue=df['Cluster'], palette='coolwarm')
plt.title('Heatmap of Waste Disposal Trends')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Cluster')
plt.show()

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
        'Total Waste Collected (kg)': reg_preds[0],
        'Average Collection Rate per Day (kg)': reg_preds[1],
        'Recycling Rate (%)': reg_preds[2]
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
