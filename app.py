import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import streamlit as st
import pandas as pd

# Load the dataset
url = 'https://raw.githubusercontent.com/Dhanush-Garrepalli/AgriAssist/main/dataset_soil_nutrients_7.csv'
df = pd.read_csv(url)

# Prepare the data
X = df[['N', 'P', 'K', 'pH','Zn', 'Fe']]
y = df['Output']  # Target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
predictions = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy}")

# For Streamlit app
st.title('Soil Fertility Analysis')
ph = st.number_input('Enter pH level:')
n = st.number_input('Enter Nitrogen level:')
p = st.number_input('Enter Phosphorus level:')
k = st.number_input('Enter Potassium level:')
zn = st.number_input('Enter Zn level:', key='zn')
fe = st.number_input('Enter Fe level:', key='fe')

# Add inputs for the rest of the features

if st.button('Analyze'):
    # Note: Add the rest of your feature inputs here in the correct order
    input_features = scaler.transform([[n, p, k, ph,zn,fe]])  # Fill in the rest of the inputs
    prediction = model.predict(input_features)
    if prediction[0] == 0:
        result = "Low Fertile"
    else:
        result = "High Fertile"
    st.write(f'The prediction is: {result}')





















