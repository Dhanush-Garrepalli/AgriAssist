import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
url = 'https://raw.githubusercontent.com/Dhanush-Garrepalli/AgriAssist/main/dataset_soil_nutrients.csv'
df = pd.read_csv(url)

# Prepare the data with the correct nutrient columns
X = df[['N', 'P', 'K', 'pH', 'Zn', 'Fe', 'EC', 'OC', 'S', 'Cu', 'Mn', 'B']]
y = df['Output']

mean_thresholds = df[['N', 'P', 'K', 'pH', 'Zn', 'Fe', 'EC', 'OC', 'S', 'Cu', 'Mn', 'B']].mean()

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

st.title('Soil Fertility Analysis')

# Create input fields for all features
input_values = {feature: st.number_input(f'Enter {feature} level:', key=feature) for feature in X.columns}

if st.button('Analyze'):
    # Prepare user input for prediction
    user_input = np.array([input_values[feature] for feature in X.columns]).reshape(1, -1)
    input_scaled = scaler.transform(user_input)
    input_pca = pca.transform(input_scaled)
    
    # Make prediction
    prediction = model.predict(input_pca)
    result = "High Fertile" if prediction[0] == 1 else "Low Fertile"
    st.write(f'The prediction is: {result}')

    # Provide feedback based on mean thresholds
    for feature in mean_thresholds.index:
        if input_values[feature] < mean_thresholds[feature]:
            feedback_message = f'{feature} level is below average, which may affect fertility.'
            if feature.startswith('N'):
                feedback_message += ' Leaves may turn yellow.'
            st.write(feedback_message)
