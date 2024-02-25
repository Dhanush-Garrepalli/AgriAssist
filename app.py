import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Assuming the setup from the previous steps remains the same

# Load the dataset
url = 'https://raw.githubusercontent.com/Dhanush-Garrepalli/AgriAssist/main/dataset_soil_nutrients.csv'
df = pd.read_csv(url)

X = df[['N', 'P', 'K', 'pH', 'Zn', 'Fe', 'EC', 'OC', 'S', 'Cu', 'Mn', 'B']]
y = df['Output']

# Split, scale, and PCA as before
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=5)  # Adjust based on earlier analysis
X_pca = pca.fit_transform(X_scaled)

# Split the PCA transformed data for training and testing
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_pca, y_train)

# Streamlit app
st.title('Soil Fertility Analysis')

# Dynamically create input fields based on the top components
input_data = []
for feature in top_features:
    input_data.append(st.number_input(f'Enter {feature} level:', key=feature))

if st.button('Analyze'):
    input_array = np.array(input_data).reshape(1, -1)
    # Scale and transform the input to match PCA transformation
    input_scaled = scaler.transform(input_array)
    input_pca = pca.transform(input_scaled)
    
    # Prediction
    prediction = model.predict(input_pca)
    result = "High Fertile" if prediction[0] == 1 else "Low Fertile"
    st.write(f'The prediction is: {result}')
