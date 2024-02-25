import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Load the dataset
url = 'https://raw.githubusercontent.com/Dhanush-Garrepalli/AgriAssist/main/dataset_soil_nutrients.csv'
df = pd.read_csv(url)

# Prepare the data with the correct nutrient columns
X = df[['N', 'P', 'K', 'pH', 'Zn', 'Fe', 'EC', 'OC', 'S', 'Cu', 'Mn', 'B']]
y = df['Output']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to identify the components that explain most variance
pca = PCA(n_components=6)
X_pca = pca.fit_transform(X_scaled)

# Train a RandomForestClassifier on the PCA-reduced dataset
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_pca, y_train)

# Calculate total importance of each feature across all PCA components
total_importance = np.abs(pca.components_).sum(axis=0)
sorted_indices = np.argsort(total_importance)[::-1]
top_features = X.columns[sorted_indices][:5].tolist()

# Streamlit app
st.title('Soil Fertility Analysis')

# Dynamically create input fields based on the top features
input_data = []
for feature in top_features:
    input_data.append(st.number_input(f'Enter {feature} level:', key=feature))

if st.button('Analyze'):
    # Prepare user input for prediction
    input_df = pd.DataFrame([input_data], columns=top_features)
    input_scaled = scaler.transform(input_df)  # Scale the user input
    input_pca = pca.transform(input_scaled)  # Transform user input with PCA
    
    # Make prediction
    prediction = model.predict(input_pca)
    result = "High Fertile" if prediction[0] == 1 else "Low Fertile"
    st.write(f'The prediction is: {result}')
