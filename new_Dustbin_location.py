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
import folium
from folium.plugins import HeatMap

# Load dataset
df = pd.read_excel("new_bin_map.xlsx")
df.rename(columns=lambda x: x.strip(), inplace=True)

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

# Define high-demand threshold
fullness_threshold = 75  # Bins filled more than 75%
collection_threshold = 3  # Bins collected more than 3 times a day

# Find locations that meet the criteria
high_demand_bins = df[(df['Fullness (%)'] > fullness_threshold) & 
                      (df['Collection Frequency'] > collection_threshold)]

#print(high_demand_bins[['Latitude', 'Longitude']])

# Create a map centered around a general location
m = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=12)

# Add heatmap layer
heat_data = list(zip(df["Latitude"], df["Longitude"], df["Fullness (%)"]))
HeatMap(heat_data).add_to(m)

# Save map to an HTML file
m.save("waste_heatmap.html")

print("Heatmap saved as waste_heatmap.html. Open this file in a browser to view.")

# Get coordinates of high-demand bins
coordinates = high_demand_bins[['Latitude', 'Longitude']].values

# Apply K-Means clustering (adjust number of clusters as needed)
kmeans = KMeans(n_clusters=5, random_state=42)  # Suggesting 5 new locations
kmeans.fit(coordinates)

# Get new bin placement locations
new_bin_locations = kmeans.cluster_centers_

print("Suggested locations for new bins:")
print(new_bin_locations)
