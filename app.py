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

# Calculate total importance of each feature across all PCA components
total_importance = np.abs(pca.components_).sum(axis=0)
sorted_indices = np.argsort(total_importance)[::-1]
top_features = X.columns[sorted_indices][:5].tolist()

# Streamlit app
st.title('Soil Fertility Analysis')

# Create a placeholder for all features with zeros
user_input = np.zeros(len(X.columns))

# Dynamically create input fields based on the top features
for i, feature in enumerate(X.columns):
    if feature in top_features:
        user_input[i] = st.number_input(f'Enter {feature} level:', key=feature)
    else:
        user_input[i] = 0  # Placeholder for features not in top_features

if st.button('Analyze'):
    # Prepare user input for prediction
    user_input = user_input.reshape(1, -1)  # Reshape to match the expected input shape
    input_scaled = scaler.transform(user_input)  # Scale the user input
    input_pca = pca.transform(input_scaled)  # Transform user input with PCA
    
    # Make prediction
    prediction = model.predict(input_pca)
    result = "High Fertile" if prediction[0] == 1 else "Low Fertile"
    st.write(f'The prediction is: {result}')

    if ph < mean_thresholds['pH']:
        st.write('pH level is below average')
    if n < mean_thresholds['N']:
        st.write('Nitrogen-deficient plants produce smaller than normal fruit, leaves, and shoots and these can develop later than normal')
    if p < mean_thresholds['P']:
        st.write('Phosphorus deficiency can cause leaves to darken and take on a dull, blue-green hue, which may lighten to pale in more extreme cases')
    if k < mean_thresholds['K']:
        st.write('Potassium deficiency in broadleaves causes leaves to turn yellow and then brown at the tips and margins and between veins.')
    if zn < mean_thresholds['Zn']:
        st.write('Zinc deficiency negatively affects plant growth.')
    if fe < mean_thresholds['Fe']:
        st.write('Fe Iron deficiency will turn leaves to yellow')
