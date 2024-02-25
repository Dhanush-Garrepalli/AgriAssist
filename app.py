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

# Calculate total importance of each feature across all PCA components
total_importance = np.abs(pca.components_).sum(axis=0)
sorted_indices = np.argsort(total_importance)[::-1]
top_features = X.columns[sorted_indices][:6].tolist()

# Train a RandomForestClassifier on the PCA-reduced dataset
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_pca, y_train)

st.title('Soil Fertility Analysis')

# Create input fields for the top 6 features only
input_values = {}
for feature in top_features:
    input_values[feature] = st.number_input(f'Enter {feature} level:', key=feature)

# Button to perform analysis
if st.button('Analyze'):
    # Prepare user input for prediction
    user_input_array = np.zeros(X.shape[1])  # Initialize an array with zeros for all features
    for i, feature in enumerate(X.columns):
        if feature in input_values:
            user_input_array[i] = input_values[feature]  # Update the array with user-provided values for top features

    user_input = user_input_array.reshape(1, -1)  # Reshape to match expected input shape
    input_scaled = scaler.transform(user_input)  # Scale the user input
    input_pca = pca.transform(input_scaled)  # Transform user input with PCA
    
    # Make prediction
    prediction = model.predict(input_pca)
    result = "High Fertile" if prediction[0] == 1 else "Low Fertile"
    st.write(f'The prediction is: {result}')

    # Provide feedback based on mean thresholds for the top features only
    for feature in top_features:
        if input_values[feature] < mean_thresholds[feature]:
            feedback_message = f'{feature} level is below average, which may affect fertility.'
            if feature.startswith('N'):
                feedback_message += 'Nitrogen-deficient plants produce smaller than normal fruit, leaves, and shoots and these can develop later than normal.'
            if feature.startswith('P'):
                feedback_message += 'Phosphorus deficiency can cause leaves to darken and take on a dull, blue-green hue, which may lighten to pale in more extreme cases.'
            if feature.startswith('K'):
                feedback_message += 'Potassium deficiency in broadleaves causes leaves to turn yellow and then brown at the tips and margins and between veins.'
            if feature.startswith('p'):
                feedback_message += 'Low PH levels makes plant growth slower'
            if feature.startswith('Z'):
                feedback_message += 'Zinc deficiency negatively affects plant growth.'
            st.write(feedback_message)
